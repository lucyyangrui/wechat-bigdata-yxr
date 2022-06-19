from transformers import AdamW, get_cosine_schedule_with_warmup


def build_optimizer(args, model, mode='pretrain', total_step=10000):
    if mode == 'pretrain':
        lr_dict = {'others':5e-4, 'bert_model':5e-5}  # 5e-5
    else:
        lr_dict = {'others':5e-4, 'bert_model':5e-5}
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = []

    for layer_name in lr_dict:
        lr = lr_dict[layer_name]
        if layer_name != 'others':
            optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                       and layer_name in n)],
                 'weight_decay': args.weight_decay,
                 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                       and layer_name in n)],
                 'weight_decay': 0.0,
                 'lr': lr}
            ]
        else:
            optimizer_grouped_parameters += [
                {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                       and not any(name in n for name in lr_dict))],
                 'weight_decay': args.weight_decay,
                 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                       and not any(name in n for name in lr_dict))],
                 'weight_decay': 0.0,
                 'lr': lr}
            ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_dict['bert_model'], eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_step * args.warm_up_ratio,
                                                num_training_steps=total_step)
    return optimizer, scheduler

