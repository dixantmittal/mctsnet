import pickle
import time
from datetime import datetime


def checkpoint(shared_model, shared_dataset, args):
    try:
        while True:
            # Save dataset
            file = open(args.data, 'wb')
            pickle.dump(list(shared_dataset), file)
            file.close()

            # Save model
            now = datetime.now().strftime("%d_%m_%H_%M")
            shared_model.save('models/checkpoint_{}.model'.format(now))

            time.sleep(10 * 60)

    except KeyboardInterrupt:
        print('exiting checkpoint')
