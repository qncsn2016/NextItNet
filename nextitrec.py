import tensorflow as tf
import data_loader_recsys
import generator_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse


# You can run it directly, first training and then evaluating
# nextitrec_generate.py can only be run when the model parameters are saved, i.e.,
#  save_path = saver.save(sess,
#                       "Data/Models/generation_model/model_nextitnet.ckpt".format(iter, numIters))
# if you are dealing very huge industry dataset, e.g.,several hundred million items, you may have memory problem during training, but it 
# be easily solved by simply changing the last layer, you do not need to calculate the cross entropy loss
# based on the whole item vector. Similarly, you can also change the last layer (use tf.nn.embedding_lookup or gather) in the prediction phrase 
# if you want to just rank the recalled items instead of all items. The current code should be okay if the item size < 5 million.



# Strongly suggest running codes on GPU with more than 10G memory!!!
# if your session data is very long e.g, >50, and you find it may not have very strong internal sequence properties, you can consider generate subsequences
def generatesubsequence(train_set):
    # create subsession only for training
    subseqtrain = []
    for i in range(len(train_set)):
        # print(x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # session lens=100 shortest subsession=5 realvalue+95 0
        for j in range(lenseq - 2):
            subseqend = seq[:len(seq) - j]
            subseqbeg = [0] * j
            subseq = np.append(subseqbeg, subseqend)
            # beginseq=padzero+subseq
            # newsubseq=pad+subseq
            subseqtrain.append(subseq)
    x_train = np.array(subseqtrain)  # list to ndarray
    del subseqtrain
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_train = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_train]
    print("generating subsessions is done!")
    return x_train


def main(datapath=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    # history_sequences_20181014_fajie_smalltest.csv
    parser.add_argument('--datapath', type=str, default='Data/Session/musicl_20.csv',
    # parser.add_argument('--datapath', type=str, default='Data/Session/user-filter-20000items-session5.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=100,
                        help='Sample generator output every x steps')
    parser.add_argument('--save_para_every', type=int, default=10000,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=True,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    args = parser.parse_args()

    if datapath:
        dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': datapath})
    else:
        dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.text_dir})
        datapath = args.text_dir

    all_samples = dl.item
    items = dl.item_dict

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_generatesubsession:
        train_set = generatesubsequence(train_set)

    model_para = {
        # if you changed the parameters here, also do not forget to change paramters in nextitrec_generate.py
        'item_size': len(items),
        'dilated_channels': 100,
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        # when you change it do not forget to change it in nextitrec_generate.py
        'dilations': [1, 2, 4],
        'kernel_size': 3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'iterations': 256,
        'is_negsample': False  # False denotes no negative sampling
    }

    print("\n-------------------------------")
    print("model: NextItRec")
    print("is_generatesubsession:", args.is_generatesubsession)
    print("train_set.shape[0]:", train_set.shape[0])
    print("train_set.shape[1]:", train_set.shape[1])
    print("dataset:", datapath)
    print("batch_size:", model_para['batch_size'])
    print("embedding_size:",  model_para['dilated_channels'])
    print("learning_rate:", model_para['learning_rate'])
    print("-------------------------------\n")

    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(model_para['is_negsample'])
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1).minimize(itemrec.loss)
    itemrec.predict_graph(model_para['is_negsample'], reuse=True)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    maxmrr5, maxmrr20, maxhit5, maxhit20, maxndcg5, maxndcg20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for iter in range(model_para['iterations']):

        # train
        train_loss = []
        batch_no_train = 0
        batch_size = model_para['batch_size']
        start = time.time()
        t1 = time.time()
        print("Iter:%d\ttotal train batch:%d" % (iter, round(len(train_set)/batch_size)))
        while (batch_no_train + 1) * batch_size < len(train_set):
            train_batch = train_set[batch_no_train * batch_size: (batch_no_train + 1) * batch_size, :]
            _, loss, results = sess.run(
                [optimizer, itemrec.loss,
                 itemrec.arg_max_prediction],
                feed_dict={
                    itemrec.itemseq_input: train_batch
                })
            train_loss.append(loss)
            batch_no_train += 1

            t3 = time.time() - start
            if t3 > 300:
                print("batch_no_train: %d, total_time: %.2f" % (batch_no_train, t3))

            if batch_no_train % 10 == 0:
                t2 = time.time()
                print("batch_no_train: %d, time:%.2fs, loss: %.4f" % (batch_no_train, t2 - t1, np.mean(train_loss)))
                t1 = time.time()

        end = time.time()
        print("train LOSS: %.4f, time: %.2fs" % (np.mean(train_loss), end - start))

        # test
        test_loss = []
        batch_no_test = 0
        formrr5, forhit5, forndcg5, formrr20, forhit20, forndcg20 = [], [], [], [], [], []
        _maxmrr5, _maxmrr20, _maxrecall5, _maxrecall20, _maxndcg5, _maxndcg20 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        start = time.time()
        print("Iter:%d total test batch:%d" % (iter, round(len(valid_set) / batch_size)))
        while (batch_no_test + 1) * batch_size < len(valid_set):
            _formrr5, _forhit5, _forndcg5, _formrr20, _forhit20, _forndcg20 = [], [], [], [], [], []
            test_batch = valid_set[batch_no_test * batch_size: (batch_no_test + 1) * batch_size, :]
            [probs], loss = sess.run(
                [[itemrec.g_probs], [itemrec.loss_test]],
                feed_dict={
                    itemrec.input_predict: test_batch
                })
            test_loss.append(loss)
            batch_no_test += 1

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
            for bi in range(probs.shape[0]):
                pred_items_5 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k)  # top_k=5
                pred_items_20 = utils.sample_top_k(probs[bi][-1], top_k=args.top_k + 15)

                true_item = item_batch[bi][-1]
                predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5)}
                pred_items_20 = {ch: i for i, ch in enumerate(pred_items_20)}

                rank_5 = predictmap_5.get(true_item)
                rank_20 = pred_items_20.get(true_item)
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

        if thisndcg5 > maxndcg5:
            maxndcg5 = thisndcg5
            maxmrr5 = thismrr5
            maxhit5 = thishit5
        if thisndcg20 > maxndcg20:
            maxndcg20 = thisndcg20
            maxmrr20 = thismrr20
            maxhit20 = thishit20


            # print("BATCH_NO: {}".format(batch_no_test))
        end = time.time()
        print("test LOSS: %.4f, time: %.2fs" % (np.mean(test_loss), end - start))

        print("\t\t\t\t\t\t\tmax_hit_5=%.4f   max_mrr_5=%.4f   max_ndcg_5=%.4f" %
              (maxhit5, maxmrr5, maxndcg5))  # 5
        print("\t\t\t\t\t\t\tmax_hit_20=%.4f  max_mrr_20=%.4f  max_ndcg_20=%.4f\n" %
              (maxhit20, maxmrr20, maxndcg20))  # 20


if __name__ == '__main__':
    data='movie5'
    t1 = time.time()
    main('Data/Session/' + data + '.csv')
    t2 = time.time()
    print("nextitnet %s time = %.2f mins" % (data, (t2 - t1) / 60))

