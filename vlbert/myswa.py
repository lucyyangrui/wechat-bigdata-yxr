import torch, json
from config import parse_args
from model_pretrain import MultiModal
from finetune import validation
from data_helper import create_dataloaders
from util import setup_device, setup_seed
from category_id_map import category_id_to_lv2id
from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == '__main__':
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    
    print('begining...')

    file_name_list = [# './save/finetune/kfold/10-fold/model_fold_9_epoch_3_f1_0.6678.bin',
                      './save/finetune/kfold/10-fold/model_fold_9_epoch_4_f1_0.6692.bin', # 
                      # './save/finetune/kfold/10-fold/model_fold_2_epoch_2_f1_0.6574.bin',
                      # './save/finetune/kfold/model_fgm_0_epoch_3_f1_0.6713.bin',
                      ]
    state_dict = []
    for fn in file_name_list:
        state_dict.append(torch.load(fn, map_location='cpu'))

    f1_model = state_dict[0]
    model_keys = f1_model['model_state_dict'].keys()

    for k in model_keys:
        tmp_weight = f1_model['model_state_dict'][k]
        for ep in range(1, len(state_dict)):
            tmp_weight += state_dict[ep]['model_state_dict'][k]
        f1_model['model_state_dict'][k] = tmp_weight / len(state_dict)

    torch.save(f1_model,  './save/finetune/kfold/10-fold/' + 'last-ckp-9.bin')

    model = MultiModal(args)
    model.load_state_dict(f1_model['model_state_dict'])
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    labels = [category_id_to_lv2id(anns[idx]['category_id']) for idx in range(len(anns))]
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    fold_n = 9
    for (f, (train_index, val_index)) in enumerate(split.split(range(100000), labels)):
        if f == fold_n:
            _, val_dataloader = create_dataloaders(args, train_index, val_index, pretrain=False)
            loss, results = validation(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            print(f"swa: loss {loss:.3f}, {results}")

