from torch.utils.data import DataLoader
from dataloaders.Weed_Carrot import CarrotWeed
from dataloaders.Weed_Carrot_nostram import CarrotWeedNS
from dataloaders.weed_Bonisugarbeet import BornsugarbeetWeed
from dataloaders.Weed_Rice import RiceWeed
from dataloaders.Weed_Sunflower import SunflowerWeed
from dataloaders.weed_Bonisugarbeet_nostream import BornsugarbeetWeedNS


def make_data_loader(args, **kwargs):

    # loading the Carrot weeds dataset
    if args.dataset == 'cweeds':
        train_set = CarrotWeed(args, base_to='train')
        val_set = CarrotWeed(args, base_to='val')
        test_set = CarrotWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Bonirob sugarbeets weeds dataset
    elif args.dataset == 'bweeds':
        train_set = BornsugarbeetWeed(args, base_to='train')
        val_set = BornsugarbeetWeed(args, base_to='val')
        test_set = BornsugarbeetWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Onion weeds dataset
    elif args.dataset == 'rweeds':
        train_set = RiceWeed(args, base_to='train')
        val_set = RiceWeed(args, base_to='val')
        test_set = RiceWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    # loading the Pea weeds dataset
    elif args.dataset == 'sweeds':
        train_set = SunflowerWeed(args, base_to='train')
        val_set = SunflowerWeed(args, base_to='val')
        test_set = SunflowerWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Pea weeds dataset
    elif args.dataset == 'svweeds':

        train_set = SunflowerWeed(args, base_to='train')
        val_set = SunflowerWeed(args, base_to='val')
        test_set = SunflowerWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


def make_data_loader_nostream(args, **kwargs):

    # loading the Carrot weeds dataset
    if args.dataset == 'cweeds':
        train_set = CarrotWeedNS(args, base_to='train')
        val_set = CarrotWeedNS(args, base_to='val')
        test_set = CarrotWeedNS(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Bonirob sugarbeets weeds dataset
    elif args.dataset == 'bweeds':
        train_set = BornsugarbeetWeedNS(args, base_to='train')
        val_set = BornsugarbeetWeedNS(args, base_to='val')
        test_set = BornsugarbeetWeedNS(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Onion weeds dataset
    elif args.dataset == 'rweeds':
        train_set = RiceWeed(args, base_to='train')
        val_set = RiceWeed(args, base_to='val')
        test_set = RiceWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    # loading the Pea weeds dataset
    elif args.dataset == 'sweeds':
        train_set = SunflowerWeed(args, base_to='train')
        val_set = SunflowerWeed(args, base_to='val')
        test_set = SunflowerWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    # loading the Pea weeds dataset
    elif args.dataset == 'svweeds':

        train_set = SunflowerWeed(args, base_to='train')
        val_set = SunflowerWeed(args, base_to='val')
        test_set = SunflowerWeed(args, base_to='test')

        num_class = train_set.NUM_CLASSES

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError