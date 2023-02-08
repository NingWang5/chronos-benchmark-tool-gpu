#!/usr/bin/env python

from utils_data import generate_data
from models import generate_forecaster
from pytorch_lightning import Trainer as Trainer_lit
import time
import numpy as np
import argparse
import os
from scipy import stats
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import onnxruntime


def torch_to_onnx(args, lit_model, test_loader, model_path):

    if not os.path.exists(model_path+'/onnx_models'):
        os.makedirs(model_path+'/onnx_models')

    import torch
    dummy_input = None
    for x, _ in test_loader:
        dummy_input = x
        break

    inputs = ['input_data']
    outputs = ['output_data']
    dynamic_axes = {'input_data': {0: 'batch'}, 'output_data':{0:'batch'}}
    onnx_path = model_path+'/onnx_models/'+str(args.horizon)+'_'+str(args.dataset)+'.onnx'
    out = torch.onnx.export(lit_model, dummy_input, onnx_path, input_names=inputs, output_names=outputs)#, dynamic_axes=dynamic_axes)

    return onnx_path

def train(args, trainer, lit_model, train_loader, test_loader, records, model_path):
    """
    train stage will record throughput.
    """

    epochs = args.training_epochs

    start_time = time.time()
    trainer.fit(lit_model, train_loader, test_loader)
    training_time = time.time() - start_time
    InferenceOptimizer.save(lit_model, model_path)
    torch_to_onnx(args, lit_model, test_loader, model_path)

    training_sample_num = epochs * len(train_loader.dataset)

    records['training_time'] = training_time
    records['training_sample_num'] = training_sample_num
    records['train_throughput'] = training_sample_num / training_time

    return lit_model


def latency(args, trainer, lit_model, train_loader, test_loader, records, model_path):
    """
    latency stage will record inference latency.
    """
    if onnxruntime.get_device() == 'CPU':
        try:
            lit_model = InferenceOptimizer.load(model_path, model=lit_model)
        except:
            trainer.fit(lit_model, train_loader, test_loader)
            InferenceOptimizer.save(lit_model, model_path)
            pass

        onnx_path = torch_to_onnx(args, lit_model, test_loader, model_path)


    latency, latency_onnx, latency_vino, latency_jit = [], [], [], []
    latency_trim_portion = 0.1
    latency_percentile = [50, 90, 95, 99]

    x = None
    for x, _ in train_loader:
        break
    input_sample = x[0].unsqueeze(0)

    # predict
    if 'torch' in args.inference_framework:
        import trts.trt as trts
        records['torch_latency'], records['torch_percentile_latency'] = trts.main(args, onnx_path, train_loader, test_loader)

    # predict with onnx
    if 'onnx' in args.inference_framework:
        onnx_model_path = os.path.join(model_path, 'onnx_'+str(args.horizon)+'_'+str(args.quantize)+'_'+str(args.dataset))
        # if onnxruntime.get_device() == 'CPU':
        
        if args.quantize:
            onnx_q_model = InferenceOptimizer.quantize(lit_model, accelerator='onnxruntime', calib_data=train_loader)
            model = onnx_q_model
        else:
            onnx_model = InferenceOptimizer.trace(lit_model, accelerator="onnxruntime", input_sample=input_sample)
            model = onnx_model
        InferenceOptimizer.save(model, onnx_model_path)
        # else:
        #     session = onnxruntime.InferenceSession(os.path.join(onnx_model_path, 'onnx_saved_model.onnx'), providers=['TensorrtExecutionProvider'])#, 'CUDAExecutionProvider'])
        #     input_name = session.get_inputs()[0].name
        #     output_name = session.get_outputs()[0].name
        #     for x, y in test_loader:
        #         x = x.numpy()
        #         st = time.time()
        #         session.run([output_name], {input_name: x})
        #         latency_onnx.append(time.time()-st)
        #     records['onnx_latency'] = stats.trim_mean(latency_onnx, latency_trim_portion)
        #     records['onnx_percentile_latency'] = np.percentile(latency_onnx, latency_percentile)

    # predict with jit
    if 'jit' in args.inference_framework:
        import torch
        jit_model = torch.jit.trace(lit_model.model, input_sample)
        model = jit_model.cuda()
        for x, y in test_loader:
            x = x.cuda()
            st = time.time()
            model(x)
            latency_jit.append(time.time()-st)
        records['jit_latency'] = stats.trim_mean(latency_jit, latency_trim_portion)
        records['jit_percentile_latency'] = np.percentile(latency_jit, latency_percentile)


def result(args, records):
    """
    print benchmark information
    """

    print(">>>>>>>>>>>>> test-run information >>>>>>>>>>>>>")
    print("Stage:", args.stage)
    print("Dataset:", args.dataset)
    print("Lookback:", args.lookback)
    print("Horizon:", args.horizon)

    if args.stage == 'train':
        print("\n>>>>>>>>>>>>> train result >>>>>>>>>>>>>")
        print("avg throughput: {}".format(records['train_throughput']))
        print(">>>>>>>>>>>>> train result >>>>>>>>>>>>>")
    elif args.stage == 'latency':
        for framework in args.inference_framework:
            print("\n>>>>>>>>>>>>> {} latency result >>>>>>>>>>>>>".format(framework))
            print("avg latency: {}ms".format(records[framework+'_latency'] * 1000))
            print("p50 latency: {}ms".format(records[framework+'_percentile_latency'][0] * 1000))
            print("p90 latency: {}ms".format(records[framework+'_percentile_latency'][1] * 1000))
            print("p95 latency: {}ms".format(records[framework+'_percentile_latency'][2] * 1000))
            print("p99 latency: {}ms".format(records[framework+'_percentile_latency'][3] * 1000))
            print(">>>>>>>>>>>>> {} latency result >>>>>>>>>>>>>".format(framework))
    elif args.stage == 'throughput':
        for framework in args.inference_framework:
            print("\n>>>>>>>>>>>>> {} throughput result >>>>>>>>>>>>>".format(framework))
            print("avg throughput: {}".format(records[framework+'_infer_throughput']))
            print(">>>>>>>>>>>>> {} throughput result >>>>>>>>>>>>>".format(framework))
    elif args.stage == 'accuracy':
        print("\n>>>>>>>>>>>>> accuracy result >>>>>>>>>>>>>")
        for metric in args.metrics:
            print("{}: {}".format(metric, records[metric]))
        print(">>>>>>>>>>>>> accuracy result >>>>>>>>>>>>>")


def main():
    # read input arguments
    # currently designed arguments
    parser = argparse.ArgumentParser(description='Benchmarking Parameters')
    parser.add_argument('-s', '--stage', type=str, default='train', metavar='',
                        help=('stage name, choose from train/latency/throughput/accuracy,'
                              ' default to "train".'))
    parser.add_argument('-d', '--dataset', type=str, default="tsinghua_electricity", metavar='',
                        help=('dataset name, choose from nyc_taxi/tsinghua_electricity/'
                              'synthetic_dataset, default to "tsinghua_electricity".'))
    parser.add_argument('-l', '--lookback', type=int, metavar='lookback', required=True,
                        help='required, the history time steps (i.e. lookback).')
    parser.add_argument('-o', '--horizon', type=int, metavar='horizon', required=True,
                        help='required, the output time steps (i.e. horizon).')

    # useful arguments which are not concluded in the currently designed pattern.
    parser.add_argument('--training_batchsize', type=int, default=32, metavar='',
                        help='batch size when training, default to 32.')
    parser.add_argument('--training_epochs', type=int, default=1, metavar='',
                        help='number of epochs when training, default to 1.')
    parser.add_argument('--inference_batchsize', type=int, default=1, metavar='',
                        help='batch size when infering, default to 1.')
    parser.add_argument('--quantize', action='store_true',
                        help='if use the quantized model to predict, default to False.')
    parser.add_argument('--inference_framework', nargs='+', default=['torch'], metavar='',
                        help=('predict without/with accelerator, choose from torch/onnx/openvino,'
                        ' default to "torch" (i.e. predict without accelerator).'))
    parser.add_argument('--quantize_type', type=str, default='pytorch_fx', metavar='',
                        help=('quantize framework, choose from pytorch_fx/pytorch_ipex/'
                              'onnxrt_qlinearops/openvino, default to "pytorch_fx".'))
    parser.add_argument('--ckpt', type=str, default='checkpoints/tcn', metavar='',
                        help=('checkpoint path of a trained model, e.g. "checkpoints/tcn",'
                              ' default to "checkpoints/tcn".'))
    parser.add_argument('--metrics', type=str, nargs='+', default=['mse', 'mae'], metavar='',
                        help=('evaluation metrics of a trained model, e.g. "mse"/"mae",'
                              ' default to "mse, mae".'))
    parser.add_argument('--normalization', action='store_false',
                        help='if to use normalization trick to alleviate distribution shift.')
    parser.add_argument('--dummy_encoder', action='store_false',
                        help='if to use dummy encoder.')
    args = parser.parse_args()
    records = vars(args)

    # anomaly detection for input arguments
    stages = ['train', 'latency', 'throughput', 'accuracy']
    datasets = ['tsinghua_electricity', 'nyc_taxi', 'synthetic_dataset', 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
    quantize_types = ['pytorch_fx', 'pytorch_ipex', 'onnxrt_qlinearops', 'openvino']
    quantize_torch_types = ['pytorch_fx', 'pytorch_ipex']

    invalidInputError(args.stage in stages,
                      f"-s/--stage argument should be one of {stages}, but get '{args.stage}'")
    invalidInputError(args.dataset in datasets,
                      (f"-d/--dataset argument should be one of {datasets},"
                       " but get '{args.dataset}'"))
    invalidInputError(args.quantize_type in quantize_types,
                      (f"--quantize_type argument should be one of {quantize_types},"
                       " but get '{args.quantize_type}'"))
    if args.quantize and 'torch' in args.inference_framework:
        invalidInputError(args.quantize_type in quantize_torch_types,
                          (f"if inference framework is 'torch', then --quantize_type"
                           " argument should be one of {quantize_torch_types},"
                           " but get '{args.quantize_type}'"))

    if 'onnx' in args.inference_framework:
        args.quantize_type = 'onnxrt_qlinearops'
    elif 'openvino' in args.inference_framework:
        args.quantize_type = 'openvino'

    path = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(path, args.ckpt)

    # generate data
    train_loader, val_loader, test_loader = generate_data(args)

    # initialize forecaster
    lit_model = generate_forecaster(args)

    # initializa Trainer
    trainer = Trainer_lit(max_epochs=args.training_epochs,
                          accelerator='gpu', 
                          devices=1)

    # running stage
    if args.stage == 'train':
        train(args, trainer, lit_model, train_loader, test_loader, records, model_path)
    elif args.stage == 'latency':
        latency(args, trainer, lit_model, train_loader, test_loader, records, model_path)

    # print results
    result(args, records)


    import pandas as pd
    output_name = os.path.join(path, 'records.csv')
    if os.path.exists(output_name):
        output = pd.read_csv(output_name)
    else:
        output = pd.DataFrame(columns=['model', 'training_processes', 'training_batchsize', 'training_epochs',
       'quantize', 'inference_framework', 'ipex', 'cores', 'lookback',
       'horizon', 'stage', 'quantize_type', 'dataset', 'framework',
       'training_time', 'inference_batchsize',
       'onnx_latency', 'onnx_percentile_latency', 'openvino_latency',
       'openvino_percentile_latency', 'torch_latency',
       'torch_percentile_latency', 'onnx_infer_throughput',
       'openvino_infer_throughput', 'torch_infer_throughput',
       'train_throughput', 'training_sample_num', 'mse', 'mae', 'info'])

    output = output.append(records, ignore_index=True)
    # print(output.columns)
    output.to_csv(output_name, index=False)


if __name__ == "__main__":
    main()


