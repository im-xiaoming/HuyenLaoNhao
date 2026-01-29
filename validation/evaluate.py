import pandas as pd
from HuyenLaoNhao.utils.feature_extraction import get_features, fuse_features_with_norm
from HuyenLaoNhao,.validation import evaluate_utils

def evaluate1(model, val_loader, device):
    model.eval()

    all_embeddings = []
    all_labels = []
    all_datanames = []
    all_indices = []
    with torch.no_grad():
        for images, labels, datanames, indices in tqdm(val_loader):

            images, labels = images.to(device), labels.to(device)

            embeddings, norms = model(images)

            flip_embeddings, flip_norms = model(torch.flip(images, dims=[3]))
            embeddings, norms = fuse_features_with_norm(
                torch.stack([embeddings, flip_embeddings], 0),
                torch.stack([norms, flip_norms], 0)
            )

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_datanames.append(datanames.cpu())
            all_indices.append(indices.cpu())

        embeddings = torch.cat(all_embeddings)
        labels = torch.cat(all_labels)
        datanames = torch.cat(all_datanames)
        indices = torch.cat(all_indices)

        dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cfp_ff": 3}
        idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}

        val_acc = []

        for name, index in dataname_to_idx.items():
            # per dataset evaluation
            emb = embeddings[datanames == index].to('cpu').numpy()
            lab = labels[datanames == index].to('cpu').numpy()
            issame = lab[0::2]

            # evaluate
            tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(emb, issame, nrof_folds=10)
            acc, best_threshold = accuracy.mean(), best_thresholds.mean()

            val_acc.append(acc)

            print(f"\t[{name}] Acc={acc:.4f}, Th={best_threshold:.4f}")

    return np.mean(val_acc)


def evaluate2(model, data_name, batch_size=256, device='gpu', load_feats=False, **args):
    path = os.path.join('features_temp')
    if not os.path.exists(path):
        os.makedirs(path)
    if not load_feats:
        feats, save_path = get_features(args['root'], model, args['model_name'], data_name, batch_size, device)
        np.save(os.path.join(path, data_name, 'feats.npy'), feats)
    else:
        feats = np.load(os.path.join(path, data_name, 'feats.npy'))
        save_path = './result/{}/{}'.format(data_name, args['model_name'])
        print('result save_path', save_path)
        os.makedirs(save_path, exist_ok=True)

    evaluate(args['root'], data_name, feats, save_path)

    df = pd.read_csv(f'{save_path}/verification_result.csv')
    r = df[['1e-06', '1e-05']].to_dict()
    r = {key: val[0] for key, val in r.items()}
    return r