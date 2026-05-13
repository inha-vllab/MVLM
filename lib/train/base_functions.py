import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

# datasets related
from lib.train.dataset import Lasot, VastTrack, Otb99_lang, Tnl2k
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': getattr(cfg.DATA.TEMPLATE, "FACTOR", None),
                                   'search': getattr(cfg.DATA.SEARCH, "FACTOR", None)}
    settings.output_sz = {'template': getattr(cfg.DATA.TEMPLATE, "SIZE", 128),
                          'search': getattr(cfg.DATA.SEARCH, "SIZE", 256)}
    settings.center_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "CENTER_JITTER", None),
                                     'search':getattr(cfg.DATA.SEARCH, "CENTER_JITTER", None)}
    settings.scale_jitter_factor = {'template': getattr(cfg.DATA.TEMPLATE, "SCALE_JITTER", None),
                                    'search': getattr(cfg.DATA.SEARCH, "SCALE_JITTER", None)}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.multi_modal_vision = getattr(cfg.DATA, "MULTI_MODAL_VISION", False)
    settings.multi_modal_language = getattr(cfg.DATA, "MULTI_MODAL_LANGUAGE", False)
    settings.use_nlp = cfg.DATA.USE_NLP
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        settings.fix_norm = True
    else:
        settings.fix_norm = False


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "TNL2K_train", "OTB99_train", "VASTTRACK"]
        if name == "LASOT":
            datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader,
                                  multi_modal_vision=settings.multi_modal_vision,
                                  multi_modal_language=settings.multi_modal_language,
                                  use_nlp=settings.use_nlp['LASOT']))
        if name == "VASTTRACK":
            datasets.append(VastTrack(settings.env.vasttrack_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=settings.multi_modal_vision,
                                      multi_modal_language=settings.multi_modal_language,
                                      use_nlp=settings.use_nlp['VASTTRACK']))
        if name == "TNL2K_train":
            datasets.append(Tnl2k(settings.env.tnl2k_dir, split=None, image_loader=image_loader,
                                  multi_modal_vision=settings.multi_modal_vision,
                                  multi_modal_language=settings.multi_modal_language,
                                  use_nlp=settings.use_nlp['TNL2K']
                                  ))
        if name == "OTB99_train":
            datasets.append(Otb99_lang(settings.env.otb99_lang_dir, split='train', image_loader=image_loader,
                                       multi_modal_vision=settings.multi_modal_vision,
                                       multi_modal_language=settings.multi_modal_language,
                                       use_nlp=settings.use_nlp['OTB99']
                                       ))

    return datasets


def build_dataloaders(cfg, settings):
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqTrackProcessing(search_area_factor=search_area_factor,
                                                          output_sz=output_sz,
                                                          center_jitter_factor=settings.center_jitter_factor,
                                                          scale_jitter_factor=settings.scale_jitter_factor,
                                                          mode='sequence',
                                                          transform=transform_train,
                                                          joint_transform=transform_joint,
                                                          multi_modal_language=settings.multi_modal_language,
                                                          settings=settings)

    # Train sampler and loader
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")

    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode,
                                            multi_modal_language=settings.multi_modal_language
                                            )

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN, "TYPE", None)
    if train_type == "peft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad]},
        ]
        for n, p in net.named_parameters():
            if "prompt" not in n:
                p.requires_grad = False

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "fft":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "prompt" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "prompt" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR / cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]

        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "text_frozen":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and "clip" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]
        for n, p in net.named_parameters():
            if ("clip" in n) or ("bert" in n):
                p.requires_grad = False
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    elif train_type == "ours":
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and "clip" not in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]
        for n, p in net.named_parameters():
            if ("clip" in n) or ("bert" in n) or ("cls_token" in n) or ("text_proj" in n):
                p.requires_grad = False
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "encoder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "encoder" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.ENCODER_MULTIPLIER,
            },
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
