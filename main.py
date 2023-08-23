import os,sys
import argparse
import pickle
import glob
#from model import trainer
from copy import deepcopy
from semi_model2 import *
from dataLoader import *
from tools import *

def main_worker(args):

    model_folder = os.path.join(args.save_path, args.save_name)  # Path for the saved models
    # Path for the saved pseudo label dic
    dic_folder = os.path.join(model_folder, 'cluster_dic')
    score_path = os.path.join(model_folder, 'cluster_score.txt')  # Path for the score file
    main_cluster = os.path.join(model_folder, 'cluster.txt')  # Path for the score file
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(dic_folder, exist_ok=True)
    score_file = open(score_path, "a+")
    main_cluster_file = open(main_cluster, "a+")

    # Define the data loader————labeled data and unlabelled data
    data_list, data_label, data_length = get_data(args.train_list, args.train_path)
    lb_data, lb_label, lb_length, ulb_data, ulb_label, ulb_length = split_ssl_data(data_list, data_label, data_length,
                                                                                   num_labels=args.lb_samples_per_class,
                                                                                   num_classes=args.n_cluster)

    stage, best_epoch, next_epoch, iteration, change = check_clustering1(score_path, args.LGL)  # Check the state of this epoch
    print(stage, best_epoch, next_epoch, iteration)

    # Define the framework
    Trainer = trainer(args, lr=args.lr, n_cluster=args.n_cluster)
    modelfiles = glob.glob('%s/model0*.model' % model_folder)  # Go for all saved model
    modelfiles.sort()

    if len(modelfiles) >= 1:  # Load the previous model
        Trainer.load_parameters(modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
    else:
        epoch = 1  # Start from the first epoch
        for items in vars(args):  # Save the parameters in args
            score_file.write('%s %s\n' % (items, vars(args)[items]))
        score_file.flush()

    if epoch == 1:  # Do clustering in the first epoch
        Trainer.load_parameters(args.init_model)  # Load the init_model
        #EER, minDCF = Trainer.eval_network(args.val_list, args.val_path)
        #print("EER %2.2f%%, minDCF %2.3f%%\n" % (EER, minDCF))
        clusterLoader, lb_trainLoader = get_dataloader(args, lb_data, lb_label, ulb_data, ulb_label, ulb_length, cluster_only=True)  # Data Loader
        print("Start stage C to %d Clustering" % (iteration+1))
        dic_label, NMI, ulbacc = Trainer.cluster_network(args, clusterLoader=clusterLoader,
                                                                         lb_trainLoader=lb_trainLoader,
                                                                         n_cluster=args.n_cluster,
                                                                         score_file=main_cluster_file, epoch=epoch)  # Do clustering
        pickle.dump(dic_label, open(dic_folder + "/label%04d.pkl" % epoch, "wb"))  # Save the pseudo labels
        print_write(type='C', text=[epoch, NMI, ulbacc], score_file=score_file)
    labelfiles = glob.glob('%s/label0*.pkl' % dic_folder)
    labelfiles.sort()
    dic_label = pickle.load(open(labelfiles[-1], "rb"))
    print("Dic %s loaded!" % labelfiles[-1])
    clusterLoader, loader_dict = get_dataloader(args, lb_data, lb_label, ulb_data, ulb_label, ulb_length, dic_label=dic_label)  # data loader with the pseduo labels
    EERs = []

    while True:
        stage, best_epoch, next_epoch, iteration, change = check_clustering1(score_path, args.LGL)  # Check the state of this epoch

        if stage == 'T':  # Classification training
            print("Start stage T to Classification training")
            if change == True:
                Trainer.thresold = 1/args.n_cluster
            print(change, Trainer.thresold)
            loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold, slectedNMI = Trainer.train_network(args, epoch=epoch,
                                                                                                      loader_dict=loader_dict,
                                                                                                      gated=True, fix=True)
            print_write(type='T', text=[epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI], score_file=score_file)

        elif stage == 'V':  # LGL training
            print("Start stage V to LGL training")
            if change == True:
                Trainer.thresold = 1/args.n_cluster
            print(change, Trainer.thresold)
            if best_epoch != None:  # LGL start from the best model from 'T' stage
                Trainer.load_parameters('%s/model0%03d.model' % (model_folder, best_epoch))  # Load the best model
            loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold, slectedNMI = Trainer.train_network(args, epoch=epoch,
                                                                                          loader_dict=loader_dict,
                                                                                          gated=False, fix=True)
            print_write(type='V', text=[epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI], score_file=score_file)

        elif stage == 'L':  # LGL training
            print("Start stage L to LGL training")

            print(change, Trainer.thresold)
            if best_epoch != None:  # LGL start from the best model from 'v' stage
                Trainer.load_parameters('%s/model0%03d.model' % (model_folder, best_epoch))  # Load the best model
            loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold, slectedNMI = Trainer.train_network(args, epoch=epoch,
                                                                                                      loader_dict=loader_dict,
                                                                                                      gated=True, fix=False)
            print_write(type='L', text=[epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI], score_file=score_file)

        elif stage == 'C':  # Clustering
            iteration += 1
            if iteration > 5:  # Maximun iteration is 3
                quit()
            print("Start stage C to %d Clustering" % (iteration))
            Trainer.load_parameters('%s/model0%03d.model' % (model_folder, best_epoch))  # Load the best model
            clusterLoader,lb_trainLoader = get_dataloader(args, lb_data, lb_label, ulb_data, ulb_label,ulb_length, cluster_only=True)  # Cluster loader
            dic_label, NMI, ulbacc = Trainer.cluster_network(args, clusterLoader=clusterLoader, lb_trainLoader=lb_trainLoader, n_cluster=args.n_cluster, score_file=main_cluster_file, epoch=epoch)  # Clustering
            epoch = next_epoch
            print_write(type='C', text=[epoch, NMI, ulbacc], score_file=score_file)
            pickle.dump(dic_label, open(dic_folder + "/label%04d.pkl" % epoch, "wb"))  # Save the pseudo label dic
            print("Dic %s loaded!" %(dic_folder + "/label%04d.pkl" % epoch))
            Trainer = trainer(args, lr=args.lr, n_cluster=args.n_cluster)  # Define the framework
            Trainer.load_parameters(args.init_model)  # Load the init_model
            #Trainer.Optim.param_groups[0]['lr'] = args.lr
            # Get new dataloader with new label dic
            clusterLoader, loader_dict = get_dataloader(args, lb_data, lb_label, ulb_data, ulb_label, ulb_length, dic_label=dic_label)

        if epoch % args.test_interval == 0 and stage != 'C':  # evaluation
            Trainer.save_parameters(model_folder + "/model%04d.model" % epoch)  # Save the model
            EER, minDCF = Trainer.eval_network(args.val_list, args.val_path)
            EERs.append(EER)
            best=EERs.index(min(EERs))
            print_write(type='E', text=[epoch, EER, minDCF, min(EERs)], score_file=score_file)

        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ours_model")
    parser.add_argument('--save_path', type=str, default="exp/exp1")
    parser.add_argument('--save_name', type=str, default='demo_onlycluster')
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--init_model', type=str, default="/media/heu_underwater/软件/code/SSLmodel/Stage2/exp/model_1024_661.model")

    parser.add_argument('--lb_samples_per_class', type=int, default=10920)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--ulb_loss_ratio', type=float,
                        default=1.0, help='weight for unsupervised loss')

    parser.add_argument('--train_list', type=str, default="/media/heu_underwater/软件/code/SSLmodel/Stage2/train_list.txt",
                        help='Path for Vox2 list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
    parser.add_argument('--val_list', type=str, default="/media/heu_underwater/Data/data/voxceleb1/veri_test2.txt",
                        help='Path for Vox_O list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--train_path', type=str,
                        default="/media/heu_underwater/Data/data/voxceleb2/train/wav", help='Path to the Vox2 set')
    parser.add_argument('--val_path', type=str, default="/media/heu_underwater/Data/data/voxceleb1/test/wav",
                        help='Path to the Vox_O set')
    parser.add_argument('--musan_path', type=str,
                        default="/media/heu_underwater/Data/data/Others/musan_split", help='Path to the musan set')
    parser.add_argument('--rir_path', type=str, default="/media/heu_underwater/Data/data/Others/RIRS_NOISES/simulated_rirs",
                        help='Path to the rir set')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')
    parser.add_argument('--n_cluster', type=int,
                        default=5994, help='Number of clusters')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=80)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--LGL', type=str2bool, default=False, help='Use LGL or baseline only')

    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    warnings.filterwarnings("ignore")
    over_write_args_from_file(args, args.c)
    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    if not torch.cuda.is_available():
        raise Exception('ONLY CPU TRAINING IS SUPPORTED')
    else:
        print('Number of GPUs:', torch.cuda.device_count())
        print('Save path:', args.save_path)
        if n_gpus == 1:
            main_worker(args)