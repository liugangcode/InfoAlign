def get_data(args, load_path, transform="pyg"):
    assert transform in [
        None,
        "fingerprint",
        "smiles",
        "pyg",
        "morphology",
        "expression",
    ]
    pretrained = args.dataset == "pretrain"
    
    if pretrained:
        assert transform == "pyg"
        from .pretrain_molecule import PretrainMoleculeDataset
        from .pretrain_context import PretrainContextDataset

        molecule = PretrainMoleculeDataset(root=load_path)
        context = PretrainContextDataset(root=load_path, pre_transform=args.threshold)
        return molecule, context

    if args.dataset.startswith("finetune"):
        data_name = args.dataset.split("-")[1]
    else:
        data_name = args.dataset

    if data_name in ["broad6k", "chembl2k", "biogenadme"]:
        assert transform in ["fingerprint", "smiles", "morphology", "expression", "pyg"]
        if transform == "pyg":
            from .prediction_molecule import PygPredictionMoleculeDataset

            return PygPredictionMoleculeDataset(name=data_name, root=load_path)
        else:
            from .prediction_molecule import PredictionMoleculeDataset

            return PredictionMoleculeDataset(
                name=data_name, root=load_path, transform=transform
            )
    
    if data_name in ["moltoxcast"]:
        from ogb.graphproppred import PygGraphPropPredDataset
        
        dataset = PygGraphPropPredDataset("ogbg-" + data_name, load_path)
        dataset.eval_metric = 'roc_auc'
        assert transform in ["fingerprint", "smiles", "pyg"]

        if transform == "pyg":
            return dataset
        else:
            from .prediction_molecule import PredictionMoleculeDataset

            return PredictionMoleculeDataset(
                name="ogbg_" + data_name, root=load_path, transform=transform
            )

    raise ValueError("Dataset {} not support yet".format(args.dataset))
