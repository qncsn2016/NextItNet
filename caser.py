import tensorflow as tf
import numpy as np
import argparse
import data_loader_recsys as data_loader
import utils
import shutil
import time
import os
import sys
import math
from text_cnn_hv import TextCNN_hv

'''
reimplementation of
Personalized Top-N Sequential Recommendation via
Convolutional Sequence Embedding
screen print(has been changed a bit so that to print(the output not that ofen
'''


def main(datapath=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Learning Rate')
    parser.add_argument('--sample_every', type=int, default=2000,
                        help='Sample generator output every x steps')
    parser.add_argument('--summary_every', type=int, default=50,
                        help='Sample generator output every x steps')
    parser.add_argument('--save_model_every', type=int, default=1500,
                        help='Save model every')
    parser.add_argument('--sample_size', type=int, default=300,
                        help='Sampled output size')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--max_epochs', type=int, default=64,
                        help='Max Epochs')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')
    # parser.add_argument('--text_dir', type=str, default='Data/generator_training_data',
    #                     help='Directory containing text files')
    parser.add_argument('--text_dir', type=str, default='Data/Session/short.csv',
                        help='Directory containing text files')
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data Directory')
    parser.add_argument('--seed', type=str,
                        default='f78c95a8-9256-4757-9a9f-213df5c6854e,1151b040-8022-4965-96d2-8a4605ce456c',
                        help='Seed for text generation')
    parser.add_argument('--sample_percentage', type=float, default=0.2,
                        help='sample_percentage from whole data, e.g.0.2= 80% training 20% testing')

    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')

    parser.add_argument('--num_filters', type=int, default=100,
                        help='Number of filters per filter size (default: 128)')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--l2_reg_lambda', type=float, default=0,
                        help='L2 regularization lambda (default: 0.0)')

    parser.add_argument("--allow_soft_placement", default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, help="Log placement of ops on devices")

    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability (default: 0.5)')
    args = parser.parse_args()

    if datapath:
        dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': datapath})
    else:
        dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.text_dir})
        datapath = args.text_dir
    # text_samples=16390600  vocab=947255  session100

    all_samples = dl.item
    items = dl.item_dict

    model_options = {
        'vocab_size': len(items),
        'residual_channels': 100,
    }

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    text_samples = all_samples[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(args.sample_percentage * float(len(text_samples)))
    x_train, x_dev = text_samples[:dev_sample_index], text_samples[dev_sample_index:]

    # create subsession only for training
    subseqtrain = []
    for i in range(len(x_train)):
        # print(x_train[i]
        seq = x_train[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 4):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain

    print("\n-------------------------------")
    print("model: caser")
    print("generating subsessions is done!")
    print("x_train.shape[0]:", x_train.shape[0])
    print("x_train.shape[1]:", x_train.shape[1])
    print("dataset:", datapath)
    print("batch_size:", args.batch_size)
    print("embedding_size:", model_options['residual_channels'])
    print("learning_rate:", args.learning_rate)
    print("-------------------------------\n")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]

    cnn = TextCNN_hv(
        sequence_length=x_train.shape[1],
        num_classes=len(items),
        vocab_size=len(items),
        embedding_size=model_options['residual_channels'],
        filter_sizes=eval(args.filter_sizes),
        num_filters=args.num_filters,
        loss_type=args.loss_type,
        l2_reg_lambda=args.l2_reg_lambda
    )

    session_conf = tf.ConfigProto(
        # allow to distribute device automatically if your assigned device is not found
        allow_soft_placement=args.allow_soft_placement,
        # whether print(or not
        log_device_placement=args.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())

    maxmrr5, maxmrr20, maxhit5, maxhit20, maxndcg5, maxndcg20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.max_epochs):
        # train
        batch_no_train = 0
        batch_size = args.batch_size
        print("Iter:%d\ttotal train batch:%d" % (epoch, round(x_train.shape[0] / batch_size)))
        train_loss = []
        start = time.time()
        t1 = time.time()
        while (batch_no_train + 1) * batch_size < x_train.shape[0]:
            text_batch = x_train[batch_no_train * batch_size: (batch_no_train + 1) * batch_size, :]
            _, loss, prediction = sess.run(
                [train_op, cnn.loss,
                 cnn.arg_max_prediction],
                feed_dict={
                    cnn.wholesession: text_batch,
                    cnn.dropout_keep_prob: args.dropout_keep_prob
                })
            train_loss.append(loss)
            batch_no_train += 1

            '''
            if batch_no_train % 100 == 0:
                t2 = time.time()
                print("batch_no_train: %d, time:%.2fs, loss: %.4f" % (batch_no_train, t2 - t1, np.mean(train_loss)))
                t1 = time.time()
            '''
        end = time.time()
        print("train LOSS: %.4f, time: %.2fs" % (np.mean(train_loss), end - start))

        # test
        test_loss = []
        batch_no_test = 0
        batch_size_test = args.batch_size
        formrr5, forhit5, forndcg5, formrr20, forhit20, forndcg20 = [], [], [], [], [], []
        _maxmrr5, _maxmrr20, _maxrecall5, _maxrecall20, _maxndcg5, _maxndcg20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        start = time.time()
        while (batch_no_test + 1) * batch_size < len(x_dev):
            _formrr5, _forhit5, _forndcg5, _formrr20, _forhit20, _forndcg20 = [], [], [], [], [], []
            test_batch = x_dev[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
            probs, loss = sess.run(
                [cnn.probs_flat, cnn.loss],
                feed_dict={
                    cnn.wholesession: test_batch,
                    cnn.dropout_keep_prob: 1.0
                })
            test_loss.append(loss)
            batch_no_test += 1
            # probs = [probs]

            batch_out = []
            for line in test_batch:
                batch_out.append(line[-1])
            rank_l, batch_predict, _recall5, _recall20, _mrr5, _mrr20, _ndcg5, _ndcg20 \
                = utils.cau_recall_mrr_org(probs, batch_out)
            forhit5.append(_recall5)
            formrr5.append(_mrr5)
            forndcg5.append(_ndcg5)
            forhit20.append(_recall20)
            formrr20.append(_mrr20)
            forndcg20.append(_ndcg20)
            '''
            for bi in range(len(probs)):
                pred_words_5 = utils.sample_top_k(probs[bi], top_k=args.top_k)  # top_k=5
                pred_words_20 = utils.sample_top_k(probs[bi], top_k=args.top_k + 15)

                true_word = text_batch[bi][-1]
                predictmap_5 = {ch: i for i, ch in enumerate(pred_words_5)}
                pred_words_20 = {ch: i for i, ch in enumerate(pred_words_20)}

                rank_5 = predictmap_5.get(true_word)
                rank_20 = pred_words_20.get(true_word)
                if rank_5 == None:
                    formrr5.append(0.0)
                    forhit5.append(0.0)
                    forndcg5.append(0.0)
                    _formrr5.append(0.0)
                    _forhit5.append(0.0)
                    _forndcg5.append(0.0)
                else:
                    MRR_5 = 1.0 / (rank_5 + 1)
                    Rec_5 = 1.0
                    ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)
                    formrr5.append(MRR_5)
                    forhit5.append(Rec_5)
                    forndcg5.append(ndcg_5)
                    _formrr5.append(MRR_5)
                    _forhit5.append(Rec_5)
                    _forndcg5.append(ndcg_5)
                if rank_20 == None:
                    formrr20.append(0.0)
                    forhit20.append(0.0)
                    forndcg20.append(0.0)
                    _formrr20.append(0.0)
                    _forhit20.append(0.0)
                    _forndcg20.append(0.0)
                else:
                    MRR_20 = 1.0 / (rank_20 + 1)
                    Rec_20 = 1.0
                    ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)
                    formrr20.append(MRR_20)
                    forhit20.append(Rec_20)
                    forndcg20.append(ndcg_20)
                    _formrr20.append(MRR_20)
                    _forhit20.append(Rec_20)
                    _forndcg20.append(ndcg_20)
            '''
            # if np.mean(_forndcg5) > _maxndcg5 or np.mean(_forndcg20) > _maxndcg20:
            if np.mean(_ndcg5) > _maxndcg5 or np.mean(_ndcg20) > _maxndcg20:
                _maxmrr5 = np.mean(_mrr5)  # (_formrr5)
                _maxrecall5 = np.mean(_recall5)  # (_forhit5)
                _maxndcg5 = np.mean(_ndcg5)  # (_forndcg5)
                _maxmrr20 = np.mean(_mrr20)  # (_formrr20)
                _maxrecall20 = np.mean(_recall20)  # (_forhit20)
                _maxndcg20 = np.mean(_ndcg20)  # (_forndcg20)

                print("\t\tin batch recall5=%.4f  mrr5=%.4f  ndcg5=%.4f" % (_maxrecall5, _maxmrr5, _maxndcg5))
                print("\t\tin batch recall20=%.4f mrr20=%.4f ndcg20=%.4f" % (_maxrecall20, _maxmrr20, _maxndcg20))

        thismrr5 = np.mean(formrr5)  # sum(formrr5) / float(len(formrr5))
        thismrr20 = np.mean(formrr20)  # (formrr20) / float(len(formrr20))
        thishit5 = np.mean(forhit5)  # sum(forhit5) / float(len(forhit5))
        thishit20 = np.mean(forhit20)  # sum(forhit20) / float(len(forhit20))
        thisndcg5 = np.mean(forndcg5)  # sum(forndcg5) / float(len(forndcg5))
        thisndcg20 = np.mean(forndcg20)  # (forndcg20) / float(len(forndcg20))

        if thisndcg5 > maxndcg5 or thisndcg20 > maxndcg20:
            maxndcg5 = thisndcg5
            maxmrr5 = thismrr5
            maxhit5 = thishit5
            maxndcg20 = thisndcg20
            maxmrr20 = thismrr20
            maxhit20 = thishit20

        end = time.time()
        print("test LOSS: %.4f, time: %.2fs" % (np.mean(test_loss), end - start))

        print("\t\t\t\t\t\t\tmax_hit_5=%.4f   max_mrr_5=%.4f   max_ndcg_5=%.4f" %
              (maxhit5, maxmrr5, maxndcg5))  # 5
        print("\t\t\t\t\t\t\tmax_hit_20=%.4f  max_mrr_20=%.4f  max_ndcg_20=%.4f\n" %
              (maxhit20, maxmrr20, maxndcg20))  # 20


if __name__ == '__main__':
    data = 'musicm_5'
    t1 = time.time()
    main('Data/Session/' + data + '.csv')
    t2 = time.time()
    print("caser %s time = %.2f mins" % (data, (t2 - t1) / 60))
