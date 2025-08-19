import json, torch
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from models.r2gen import R2GenModel
from modules.metrics import compute_scores
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
import nltk




class Args:
    # Paths
    image_dir = 'data/iu_xray/images/'
    ann_path = 'data/iu_xray/annotation.json'
    vocab_path = 'data/iu_xray/vocab.pkl'

    # Data
    dataset_name = 'iu_xray'
    max_seq_length = 60
    threshold = 3
    num_workers = 0
    batch_size = 1

    # Visual extractor
    visual_extractor = 'resnet101'
    visual_extractor_pretrained = True

    # Transformer settings
    d_model = 512
    d_ff = 512
    d_vf = 2048
    num_heads = 8
    num_layers = 3
    dropout = 0.1
    logit_layers = 1
    bos_idx = 0
    eos_idx = 0
    pad_idx = 0
    use_bn = 0
    drop_prob_lm = 0.5

    # Relational Memory (not used but required)
    rm_num_slots = 3
    rm_num_heads = 8
    rm_d_model = 512

    # Sampling (for inference)
    sample_method = 'beam_search'
    beam_size = 3
    temperature = 1.0
    sample_n = 1
    group_size = 1
    output_logsoftmax = 1
    decoding_constraint = 0
    block_trigrams = 1

    # Dummy training stuff
    n_gpu = 1
    epochs = 1
    save_dir = '../'
    record_dir = '../'
    save_period = 1
    monitor_mode = 'max'
    monitor_metric = 'BLEU_4'
    early_stop = 50

    # Optimizer placeholders
    optim = 'Adam'
    lr_ve = 5e-5
    lr_ed = 1e-4
    weight_decay = 5e-5
    amsgrad = True
    lr_scheduler = 'StepLR'
    step_size = 50
    gamma = 0.1

    # Misc
    seed = 1111
    resume = None

args = Args()

# Tokenizer using saved vocab
tokenizer = Tokenizer(args)

# Prepare test data
test_loader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = R2GenModel(args, tokenizer).to(device)

# Load pretrained checkpoint
ckpt = torch.load('checkpoints/model_iu_xray.pth', map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()
all_gts = []
all_res = []
all_image_ids = []

with torch.no_grad():
    for (image_ids, images, reports_ids, reports_masks) in test_loader:
        images = images.to(device)
        outputs = model(images, mode='sample')  # shape (batch_size, max_seq_length)
        # Decode predicted reports
        decoded = model.tokenizer.decode_batch(outputs.cpu().numpy())
        # Decode ground-truth reports (skip start token)
        gts = model.tokenizer.decode_batch(reports_ids[:,1:].cpu().numpy())
        # Accumulate results
        all_res.extend(decoded)
        all_gts.extend(gts)
        all_image_ids.extend(image_ids)

results = []
for img_id, hyp in zip(all_image_ids, all_res):
    results.append({
        "image_id": img_id,
        "caption": hyp
    })

with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Generated captions saved to results.json.")





