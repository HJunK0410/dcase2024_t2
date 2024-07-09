import argparse

######################################################################################################
# need to split arguments for different optimizers in detail
# also need to split dropout, etc..

def args_for_data(parser):
    parser.add_argument('--sr', type=int, default=16000, help='sampling rate')
    parser.add_argument('--train_whole', type=str, default='./processed_data/train.csv')
    parser.add_argument('--train_machine', type=str, default='./processed_data/train_machine.csv')
    parser.add_argument('--test_machine', type=str, default='./processed_data/test_machine.csv')
    parser.add_argument('--train_att', type=str, default='./processed_data/train_att.csv')
    parser.add_argument('--test_att', type=str, default='./processed_data/test_att.csv')
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--result_path', type=str, default='./cat_emb/result')
    parser.add_argument('--result_file', type=str, default='tm')

def args_for_train(parser):
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--epochs', type=int, default=30, help='max epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')   
    parser.add_argument('--lr_decay', type=float, default=0.9, help='scheduler learning rate decay param')
    
def args_for_classifier(parser):
    parser.add_argument('--nhead', type=int, default=8, help='num_heads')
    parser.add_argument('--n_layers', type=int, default=6, help='num_layers')
    parser.add_argument('--dim_ff', type=int, default=2048, help='dim_feedforward')
    parser.add_argument('--drop_p', type=float, default=0.1, help='dropout')
    parser.add_argument('--drop_p2', type=float, default=0.1, help='dropout for fc')

def args_for_arcface(parser):
    parser.add_argument('--s', type=float, default=30.0)
    parser.add_argument('--m', type=float, default=0.5)
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=302, type=int)

    parser.add_argument('--pretrained_model_waveform', type=str, default="facebook/wav2vec2-base-960h", help="pretrained model name for waveform input")
    parser.add_argument('--pretrained_model_spectrogram', type=str, default='MIT/ast-finetuned-audioset-10-10-0.4593', help='pretrained model name for spectrogram input')
    
    args_for_data(parser)
    args_for_train(parser)
    args_for_classifier(parser)
    args_for_arcface(parser)
    
    args = parser.parse_args()
    return args