# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:14:05 2021

@author: WLL
"""

import argparse
import datetime
import os
import time
import json

import torch
import torch.nn as nn
from rouge import Rouge

from GAT import GAT
from Tester import SLTester
from Example import ExampleSet
from tools import utils
from tools.logger import *

def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path))

    return model



def run_test(model, dataset, loader, model_name, hps):
    test_dir = os.path.join(hps.save_root, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1])
        resfile = open(log_dir, "w")
        logger.info("[INFO] Write the Evaluation into %s", log_dir)

    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()

    iter_start_time=time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps.m, limited=hps.limited, test_dir=test_dir)

        for i, (G, index) in enumerate(loader):
            if hps.cuda:
                G.to(torch.device("cuda"))
            tester.evaluation(G, index, dataset, blocking=hps.blocking)

    running_avg_loss = tester.running_avg_loss

    if hps.save_label:
        # save label and do not calculate rouge
        json.dump(tester.extractLabel, resfile)
        tester.SaveDecodeFile()
        logger.info('   | end of test | time: {:5.2f}s | '.format((time.time() - iter_start_time)))
        return

    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hyps is selected!")
        sys.exit(1)

    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
        else:
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
    else:
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    tester.SaveDecodeFile()
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time),float(running_avg_loss)))



def main():
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')
    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
    # Important settings
    parser.add_argument('--model', type=str, default="HSumGraph", help="model structure")
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')
     # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration ')
    
    
    parser.add_argument('in_dim',type=int,default=768,help='the dim of sentence-bert embedding')
    parser.add_argument('hidden_dim',type=int,default=96,help='the dim of output of multi-GAT')
    parser.add_argument('out_dim',type=int,default=768,help='the dim of final output of GAT')
    parser.add_argument('num_heads',type=int,default=8,help='the number of heads in multi-GAT')

    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # File paths
    DATA_FILE = os.path.join(args.data_dir, "test.label.jsonl")
    LOG_PATH = args.log_root
    # train_log setting
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "test_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    hps = args
    logger.info(hps)
    if hps.model == "HSG":
        model = GAT(hps.in_dim,hps.hidden_dim,hps.out_dim,hps.num_heads)
        logger.info("[MODEL] HeterSumGraph ")
        dataset = ExampleSet(DATA_FILE)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32,collate_fn=graph_collate_fn)
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")
    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    logger.info("[INFO] Decoding...")
    run_test(model, dataset, loader, hps.test_model, hps)
    
if __name__ == '__main__':
    main()
     

















































