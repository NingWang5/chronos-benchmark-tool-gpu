
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.data import get_public_dataset, gen_synthetic_data


def generate_data(args):
    """
    Generate dataset for training or inference.

    Args:
        args: is a ArgumentParser instance, inluding users input arguments.

    Returns:
        train_loader: is a dataset used to train.
        test_loader: is a dataset used to inference.
    """

    # read data
    if args.dataset == 'tsinghua_electricity':
        tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='tsinghua_electricity',
                                                                   with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)
    elif args.dataset == 'nyc_taxi':
        tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='nyc_taxi',
                                                                   with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)
    elif args.dataset == 'synthetic_dataset':
        tsdata_train, tsdata_val, tsdata_test = gen_synthetic_data(with_split=True,
                                                                   val_ratio=0.1,
                                                                   test_ratio=0.2)
    else:
        import pandas as pd
        from bigdl.chronos.data.tsdataset import TSDataset 
        df = pd.read_csv("/root/wangning/dataset/"+args.dataset+".csv", parse_dates=["date"])
        tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="date",
                                                                        target_col=['HUFL', 'HULL', 'MUFL', 'MULL',
                                                                                    'LUFL', 'LULL', "OT"],
                                                                        with_split=True, val_ratio=0.1, test_ratio=0.2)
        # for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        #     tsdata.gen_dt_feature()

    # preprocessing data
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute(mode="last")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))

    # transfer to tensorflow or torch dataset
    add_args = {}
    train_loader = tsdata_train.to_torch_data_loader(batch_size=args.training_batchsize,
                                                        roll=True,
                                                        lookback=args.lookback,
                                                        horizon=args.horizon,
                                                        **add_args)
    val_loader = tsdata_val.to_torch_data_loader(batch_size=args.training_batchsize,
                                                    roll=True,
                                                    lookback=args.lookback,
                                                    horizon=args.horizon,
                                                    **add_args)
    test_loader = tsdata_test.to_torch_data_loader(batch_size=args.inference_batchsize,
                                                    roll=True,
                                                    lookback=args.lookback,
                                                    horizon=args.horizon,
                                                    **add_args)

    return train_loader, val_loader, test_loader

