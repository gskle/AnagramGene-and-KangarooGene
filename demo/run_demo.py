"""Run three plant-model prediction examples using the published functions."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import time

START = time.perf_counter()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import google.protobuf

ROOT = Path(__file__).resolve().parent

def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, str(ROOT.parent / relative))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def load_data(path):
    df = pd.read_csv(str(path), dtype={'gene': str, 'tx_rep': str, 'spe': str})
    required = ['gene', 'tx_rep', 'spe', 'whole_seq_real', 'whole_seq_false']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError('Missing input columns: ' + ', '.join(sorted(missing)))
    if df.empty:
        raise ValueError('Input CSV must contain at least one row.')
    for col in ['gene', 'tx_rep', 'spe']:
        if df[col].isna().any() or df[col].str.strip().eq('').any():
            raise ValueError(col + ' must contain nonempty identifiers.')
    for col in ['whole_seq_real', 'whole_seq_false']:
        good = df[col].str.fullmatch('[ACGT]+').fillna(False)
        good &= df[col].str.len().le(10000)
        if not good.all():
            raise ValueError(col + ' must contain 1-10000 uppercase A/C/G/T bases.')
    return df

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shuffle-csv', type=Path, default=ROOT / 'data/shuffle.csv')
    parser.add_argument('--deletion-csv', type=Path, default=ROOT / 'data/deletion.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--check', action='store_true', help='Compare the bundled demo against reference predictions.')
    args = parser.parse_args()
    if args.check and (args.shuffle_csv.resolve() != (ROOT / 'data/shuffle.csv').resolve() or args.deletion_csv.resolve() != (ROOT / 'data/deletion.csv').resolve()):
        parser.error('--check is only available for the bundled inputs.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shuffle = load_data(args.shuffle_csv)
    deletion = load_data(args.deletion_csv)
    pi = module('demo_pi', '02_anagram_pi/predict_and_evaluate.py')
    si = module('demo_si', '04_anagram_si/predict_on_deleted_sequences.py')
    kg = module('demo_kg', '05_kangaroo/train_plant_deletion.py')
    timings = {}
    tasks = [('pi', shuffle, ['left_out', 'right_out']),
             ('si', shuffle, ['pred_out', 'pred_negout']),
             ('kangaroo', deletion, ['pos_score', 'neg_score'])]
    for name, df, columns in tasks:
        started = time.perf_counter()
        model = tf.keras.models.load_model(str(ROOT / 'models' / name), compile=False)
        loaded = time.perf_counter()
        if name == 'pi':
            first, second = pi.predict_pairwise(model, df, batch_size=1, maxlen=10000)
        elif name == 'si':
            first, second = si.predict_fold(model, df, maxlen=10000, batch_size=1)
        else:
            first = kg.predict_scores(model, df, 'whole_seq_real', batch_size=1)
            second = kg.predict_scores(model, df, 'whole_seq_false', batch_size=1)
        scores = np.column_stack([first, second])
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise RuntimeError('Invalid probabilities from ' + name)
        result = df[['gene', 'tx_rep', 'spe']].copy()
        result[columns[0]] = first
        result[columns[1]] = second
        result.to_csv(str(args.output_dir / (name + '_predictions.csv')), index=False, float_format='%.9g')
        if args.check:
            expected = pd.read_csv(str(ROOT / 'expected' / (name + '_predictions.csv')))
            if not result[['gene', 'tx_rep', 'spe']].reset_index(drop=True).equals(expected[['gene', 'tx_rep', 'spe']]):
                raise AssertionError(name + ' reference row identifiers do not match.')
            np.testing.assert_allclose(scores, expected[columns].values, rtol=1e-4, atol=1e-5)
        ended = time.perf_counter()
        timings[name] = {'rows': len(df), 'model_load_seconds': loaded-started,
                         'prediction_and_output_seconds': ended-loaded, 'total_seconds': ended-started}
        print('{}: {} rows; {:.2f} seconds{}'.format(name, len(df), ended-started, '; reference check passed' if args.check else ''), flush=True)
        tf.keras.backend.clear_session()
    report = {'python': platform.python_version(), 'tensorflow': tf.__version__,
              'keras': keras.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
              'protobuf': google.protobuf.__version__, 'platform': platform.platform(),
              'device': 'CPU', 'intra_op_threads': os.environ['TF_NUM_INTRAOP_THREADS'],
              'inter_op_threads': os.environ['TF_NUM_INTEROP_THREADS'], 'batch_size': 1,
              'models': timings, 'total_seconds_including_imports': time.perf_counter()-START,
              'peak_process_memory_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'reference_check': 'passed' if args.check else 'not_requested'}
    with (args.output_dir / 'runtime.json').open('w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
