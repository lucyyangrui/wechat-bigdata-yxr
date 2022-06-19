import os
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model_pretrain import MultiModal
import transformers
transformers.logging.set_verbosity_error()   # 忽略warning信息


def inference(model, dataloader, model_file=None, save_path=None, is_prt_result=False):
    print('loading model...', model_file)
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    pred_prob = []
    with torch.no_grad():
        for batch in dataloader:
            pred_label_id, pred = model(task=['tag'], text_input=batch['title_input'], text_mask=batch['title_mask'],
                                        video_feature=batch['frame_input'], video_mask=batch['frame_mask'],
                                        inference=True)
            predictions.extend(pred_label_id.cpu().numpy())
            pred_prob.append(torch.softmax(pred, -1).cpu().numpy())

    if is_prt_result:
        # 4. dump results
        with open(save_path, 'w') as f:
            for pred_label_id, ann in zip(predictions, dataset.anns):
                video_id = ann['id']
                category_id = lv2id_to_category_id(pred_label_id)
                f.write(f'{video_id},{category_id}\n')

    return pred_prob


if __name__ == '__main__':
    args = parse_args()
    model = MultiModal(args)

    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    print('>>> first starting, total: 3')
    final_res = None
    for fold in range(3):
        best_model_path = os.path.join('./save/finetune/last-ckp/3/', f'last-ckp-{fold}.bin')
        pred_prob = inference(model, dataloader, best_model_path)

        res = np.vstack(pred_prob)

        if final_res is None:
            final_res = res
        else:
            final_res += res
    if final_res is not None:
        final_res /= 3

    print('>>> second starting, total: 5')
    final_res_tmp1 = None
    for fold in range(5):
        best_model_path = os.path.join('./save/finetune/last-ckp/5fold+alldata/', f'last-ckp-{fold}')
        pred_prob = inference(model, dataloader, best_model_path)

        res = np.vstack(pred_prob)

        if final_res_tmp1 is None:
            final_res_tmp1 = res
        else:
            final_res_tmp1 += res

        if fold == 4:
            final_res_tmp1 /= 5
        elif fold == 5:
            final_res_tmp1 /= 2

    print('>>> third starting, total: 0')
    final_res_tmp2 = None
    for fold in range(0):
        best_model_path = os.path.join('./save/finetune/last-ckp/', f'last-ckp-{fold}.bin')
        pred_prob = inference(model, dataloader, best_model_path)

        res = np.vstack(pred_prob)

        if final_res_tmp2 is None:
            final_res_tmp2 = res
        else:
            final_res_tmp2 += res
    if final_res_tmp2 is not None:
        final_res_tmp2 /= 4
    
    final_res += final_res_tmp1
    # final_res += final_res_tmp2
    final_res /= 2

    # final_res /= args.fold_num  # [bs, cls_num]

    final_res.tolist()

    print('>>> combining ... ...')

    predictions = np.argmax(final_res, axis=-1)  # [bs, 1]

    print('>>> saving ...')
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
