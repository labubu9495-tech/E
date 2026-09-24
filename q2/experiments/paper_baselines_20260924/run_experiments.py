"""Independent paper-model experiments using the audited common data protocol."""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from data import dump_json, sha256
from train import load_cache, training_batch, evaluate, SELECT
from paper_models import make_model, restoration_losses


def supervised(logits, regression, labels, targets):
    return F.cross_entropy(logits, labels) + F.huber_loss(regression, targets)


def full_inputs(data, indices):
    ix = torch.as_tensor(indices, device=data['audio'].device)
    return (data['text_bank'][0, ix].float(), data['audio'][ix], data['vision'][ix],
            data['sequence'][ix], data['observed'][ix])


def train_step(model, name, data, indices, rng):
    inputs, y, targets = training_batch(data, indices, rng, True)
    if name == 'mult':
        logits, reg = model(*inputs)
        loss = supervised(logits, reg, y, targets)
        return loss, dict(prediction=loss.detach(), complete=loss.detach()*0,
                          reconstruction=loss.detach()*0, attraction=loss.detach()*0)
    complete = full_inputs(data, indices)
    if name == 'emt_dlfr':
        logits, reg, missing_aux = model(*inputs, return_aux=True)
        cl, cr, complete_aux = model(*complete, return_aux=True)
        recon, attra = restoration_losses(missing_aux, complete_aux, complete, inputs[4])
    else:
        logits, reg = model(*inputs)
        cl, cr = model(*complete)
        recon = attra = reg.new_zeros(())
    pred = supervised(logits, reg, y, targets)
    cpred = supervised(cl, cr, y, targets)
    # Keep both supervised branches in the ablation so that only the two
    # restoration penalties differ, not access to train labels/full views.
    return pred + cpred + recon + attra, dict(prediction=pred.detach(), complete=cpred.detach(),
        reconstruction=recon.detach(), attraction=attra.detach())


def load_data(device):
    print('Loading audited train/valid caches...', flush=True)
    train, valid = load_cache('train', device), load_cache('valid', device)
    masks = {k: torch.from_numpy(v).to(device) for k, v in dict(np.load(ROOT/'cache/valid_conditions.npz')).items()}
    assert len(train['labels']) == 3395 and len(valid['labels']) == 728
    assert set(SELECT) <= set(masks)
    return train, valid, masks


def model_kwargs():
    with np.load(ROOT/'assets/normalization.npz') as stats:
        return dict(class_prior=stats['class_prior'].tolist(), target_median=float(stats['target_median']))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smoke(name, device, train, valid, masks):
    seed_all(17001)
    model = make_model(name, **model_kwargs()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    indices = np.arange(24)
    inputs = full_inputs(train, indices)
    labels, targets = train['labels'][:24], train['targets'][:24]
    # Eval mode disables dropout/BN updates but gradients are enabled: this
    # is a wiring/fit check on a fixed training minibatch, never model selection.
    model.eval()
    # cuDNN LSTM must retain training reserves for backward, even in this
    # deterministic fit check (single-layer LSTMs have no internal dropout).
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.train()
    losses = []
    started = time.perf_counter()
    for step in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits, reg = model(*inputs)
        loss = supervised(logits, reg, labels, targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite tiny-batch loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert np.mean(losses[-5:]) < .65 * np.mean(losses[:5]), (name, losses[:5], losses[-5:])
    # Exercise the real objective including both restored levels.
    model.train()
    loss, terms = train_step(model, name, train, np.arange(64), np.random.default_rng(99))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    scores, _ = evaluate(model, valid, masks, names=['clean'], batch=128)
    result = dict(name=name, tiny_batch_losses=losses, seconds=time.perf_counter()-started,
        loss_terms={k: float(v) for k,v in terms.items()}, clean_check=scores['clean'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
        parameters=sum(p.numel() for p in model.parameters()), not_for_model_selection=True)
    dump_json(HERE/'smoke'/f'{name}.json', result)
    print(f'Smoke passed: {name}, loss {losses[0]:.4f} -> {losses[-1]:.4f}, seconds={result["seconds"]:.1f}', flush=True)


def run(name, seed, device, train, valid, masks, protocol):
    out = HERE/'runs'/f'{name}_seed{seed}'
    signature = {p.name: sha256(p) for p in [HERE/'paper_models.py', HERE/'run_experiments.py', HERE/'protocol.json', ROOT/'data.py', ROOT/'train.py']}
    if (out/'metrics.json').exists():
        previous = json.loads((out/'config.json').read_text())
        if previous['source_sha256'] != signature:
            raise RuntimeError('Completed run has different source/protocol; use a new experiment directory')
        print('Skip completed', out.name, flush=True)
        return
    if (out/'history.json').exists():
        raise RuntimeError(f'Incomplete run exists at {out}; archive it explicitly before restarting')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    kwargs = model_kwargs()
    model = make_model(name, **kwargs).to(device)
    cfg = protocol['models'][name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    dump_json(out/'config.json', dict(name=name, seed=seed, model_args=kwargs, settings=cfg,
        protocol=protocol, source_sha256=signature, device=str(device), torch_version=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters())))
    shuffle, masking = np.random.default_rng(seed), np.random.default_rng(seed+10000)
    history, best, bad = [], (-float('inf'), -float('inf')), 0
    start = time.perf_counter()
    for epoch in range(1, protocol['max_epochs']+1):
        model.train()
        losses, terms = [], []
        order = shuffle.permutation(len(train['labels']))
        epoch_start = time.perf_counter()
        for s in range(0, len(order), protocol['batch_size']):
            indices = order[s:s+protocol['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss, components = train_step(model, name, train, indices, masking)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at {name}/{seed}/{epoch}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            terms.append({k:float(v) for k,v in components.items()})
        scores, _ = evaluate(model, valid, masks, names=SELECT, batch=128)
        score = float(np.mean([scores[c]['macro_f1'] for c in SELECT]))
        mae = float(np.mean([scores[c]['mae'] for c in SELECT]))
        row = dict(epoch=epoch, loss=float(np.mean(losses)),
            loss_components={k: float(np.mean([t[k] for t in terms])) for k in terms[0]},
            clean=scores['clean'], selection_macro_f1=score, selection_mae=mae,
            epoch_seconds=time.perf_counter()-epoch_start, elapsed_seconds=time.perf_counter()-start)
        history.append(row)
        if (score, -mae) > best:
            best, bad = (score, -mae), 0
            torch.save(dict(name=name, model_args=kwargs, state_dict=model.state_dict(), epoch=epoch,
                            selection=row, source_sha256=signature), out/'best.pt')
        else:
            bad += 1
        dump_json(out/'history.json', history)
        print(f'{out.name} epoch={epoch} loss={row["loss"]:.4f} cleanF1={scores["clean"]["macro_f1"]:.4f} selectF1={score:.4f} MAE={mae:.4f} epoch_s={row["epoch_seconds"]:.1f}', flush=True)
        if bad >= protocol['patience']:
            break
    checkpoint = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    scores, predictions = evaluate(model, valid, masks, collect=True, batch=128)
    np.savez(out/'validation_predictions.npz', ids=valid['ids'], labels=valid['labels'].cpu().numpy(),
             targets=valid['targets'].cpu().numpy(), **predictions)
    dump_json(out/'metrics.json', dict(config=name, seed=seed, best_epoch=checkpoint['epoch'],
        train_seconds=time.perf_counter()-start, selection_macro_f1=best[0], selection_mae=-best[1],
        checkpoint_sha256=sha256(out/'best.pt'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
        conditions=scores))
    print('Completed', out.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mult','emt_dlfr','emt_no_restore'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17,29,43])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    train, valid, masks = load_data(args.device)
    if args.smoke:
        smoke(args.model, args.device, train, valid, masks)
    else:
        protocol = json.loads((HERE/'protocol.json').read_text())
        for seed in args.seeds:
            run(args.model, seed, args.device, train, valid, masks, protocol)


if __name__ == '__main__':
    main()
