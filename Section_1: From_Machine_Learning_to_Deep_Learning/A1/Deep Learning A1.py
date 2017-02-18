# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import os, sys, tarfile, random, pickle, hashlib
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
from PIL import Image

'''
First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts
on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the
testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.
'''

# plt.ion()  # Uncomment to turn Interactive on

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

'''
Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.
'''

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

'''
PROBLEM 1
Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character
A through J rendered in a different font. Display a sample of the images that we just downloaded.
Hint: you can use the package IPython.display.
'''


def plot_sample(folders_dir, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle("Sample Plot", fontsize=16, fontweight='bold')
    fig_gs = gs.GridSpec(len(folders_dir), sample_size)
    for image_folder in folders_dir:
        image_links = os.listdir(image_folder)
        image_link_samples = random.sample(image_links, sample_size)
        image_full_link_sample = [os.path.join(image_folder, image_link_sample)
                                  for image_link_sample in image_link_samples]
        for image_full_link in image_full_link_sample:
            ax = fig.add_subplot(fig_gs[folders_dir.index(image_folder), image_full_link_sample.index(image_full_link)])
            ax.imshow(plt.imread(image_full_link))
            ax.set_axis_off()
    fig.savefig(title)


plot_sample(train_folders, 10, 'plots/Sample of Training Data P1')
plot_sample(test_folders, 10, 'plots/Sample of Test Data P1')


'''
Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to
fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently.
Later we'll merge them into a single dataset of manageable size. We'll convert the entire dataset into a 3D array
(image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5
to make training easier down the road. A few images might not be readable, we'll just skip them.
'''

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

'''
PROBLEM 2
Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray.
Hint: you can use matplotlib.pyplot.
'''


def plot_sample_pickled(pickled_files_dir, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle("Sample Plot", fontsize=16, fontweight='bold')
    fig_gs = gs.GridSpec(len(pickled_files_dir), sample_size)
    for pickle_file in pickled_files_dir:
        #  load pickled data
        try:
            img_data = pickle.load(open(pickle_file, "rb"))
        except (OSError, IOError) as e:
            img_data = 3
            pickle.dump(img_data, open(pickle_file, "wb"))

        image_sample_inds = random.sample(range(img_data.shape[0]), sample_size)
        for image_sample_ind in image_sample_inds:
            ax = fig.add_subplot(fig_gs[pickled_files_dir.index(pickle_file),
                                        image_sample_inds.index(image_sample_ind)])
            ax.imshow(img_data[image_sample_ind, :, :])
            ax.set_axis_off()
    fig.savefig(title)


plot_sample_pickled(train_datasets, 10, 'plots/Sample of Training Data P2')
plot_sample_pickled(test_datasets, 10, 'plots/Sample of Test Data P2')

'''
PROBLEM 3
Another check: we expect the data to be balanced across classes. Verify that.
'''

list_of_sizes = [pickle.load(open(pickled_file, "rb")).shape[0] for pickled_file in train_datasets]
list_of_sizes = np.array(list_of_sizes)

# now to plot the figure...
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Sample Sizes for Each Label", fontsize=16, fontweight='bold')
ax = fig.add_subplot(1, 1, 1)
ax.bar(range(len(list_of_sizes)), list_of_sizes)
ax.set_xlabel("Label")
ax.set_ylabel("Sample Size")
ax.set_xticks(range(len(list_of_sizes)))
ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])  # Can use chr(label[i]+ord('A')) trick as well
i = -0.25
for val in list_of_sizes:
    ax.annotate(str(val), xy=(i, val))
    i += 1
fig.savefig("plots/Sample Sizes for Each Label P3")


'''
Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all
in memory, and you can tune train_size as needed. The labels will be stored into a separate array of integers
0 through 9. Also create a validation dataset for hyperparameter tuning.
'''


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

'''
Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test
distributions to match.
'''

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

'''
PROBLEM 4
Convince yourself that the data is still good after shuffling!
'''


def plot_sample_shuffled(dataset, label, sample_size, title=None):
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle("Sample Plot", fontsize=16, fontweight='bold')
    fig_gs = gs.GridSpec(1, sample_size)
    for i in range(sample_size):
        ax = fig.add_subplot(fig_gs[0, i])
        ax.imshow(dataset[i, :, :])
        ax.set_title('{}'.format(chr(label[i]+ord('A'))))
        ax.set_axis_off()
    fig.savefig(title)


plot_sample_shuffled(train_dataset, train_labels, 10, 'plots/Sample of Training Data P4')
plot_sample_shuffled(test_dataset, test_labels, 10, 'plots/Sample of Test Data P4')

# now to plot the figure...
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Sample Sizes for Each Label", fontsize=16, fontweight='bold')
ax = fig.add_subplot(1, 1, 1)
ax.hist(train_labels)
ax.set_xlabel("Label")
ax.set_ylabel("Sample Size")
ax.set_xticks(range(len(list_of_sizes)))
ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])  # Can use chr(label[i]+ord('A')) trick as well
fig.savefig("plots/Sample Sizes for Each Label")

'''
Finally, let's save the data for later reuse:
'''

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()

except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

'''
Problem 5
By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained
in the validation and test set! Overlap between training and test can skew the results if you expect to use your model
in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when
you use it. Measure how much overlap there is between training, validation and test samples.

Optional questions:
What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
'''
# Reference：https://github.com/Arn-O/udacity-deep-learning/blob/master/1_notmnist.ipynb
# and https://github.com/hankcs/udacity-deep-learning/blob/master/1_notmnist.py


def find_duplicates(dataset_1, dataset_2):
    dataset_1_hash = np.array([hashlib.sha256(img).hexdigest() for img in dataset_1])
    dataset_2_hash = np.array([hashlib.sha256(img).hexdigest() for img in dataset_2])
    hash_of_overlap = set(dataset_1_hash).intersection(set(dataset_2_hash))
    d1_duplicates = np.hstack([np.where(dataset_1_hash == curr_hash)[0] for curr_hash in hash_of_overlap])
    d2_duplicates = np.hstack([np.where(dataset_2_hash == curr_hash)[0] for curr_hash in hash_of_overlap])
    return d1_duplicates, d2_duplicates


#  load pickled data
try:
    data = pickle.load(open("notMNIST.pickle", "rb"))
except (OSError, IOError) as e:
    data = 3
    pickle.dump(data, open("notMNIST.pickle", "wb"))


#  compare training data with validation and test data
train_set = data['train_dataset']
train_labels = data['train_labels']
valid_set = data['valid_dataset']
valid_labels = data['valid_labels']
test_set = data['test_dataset']
test_labels = data['test_labels']


def delete_duplicates(dataset_1, dataset_2, dataset_2_labels):
    #  deletes data from dataset_2 that are in dataset_1
    dups = find_duplicates(dataset_1, dataset_2)
    return np.delete(dataset_2, dups[1], 0), np.delete(dataset_2_labels, dups[1], 0)


print("there are {} images from training set in validation set"
      .format(len(find_duplicates(train_set, valid_set)[1])))
print("there are {} images from training set in test set"
      .format(len(find_duplicates(train_set, test_set)[1])))

cleaned_test_set, cleaned_test_labels = delete_duplicates(train_set, test_set, test_labels)
cleaned_valid_set, cleaned_valid_labels = delete_duplicates(train_set, valid_set, valid_labels)
print("Test set went from {0} to {1}".format(test_set.shape, cleaned_test_set.shape))
print("Test set went from {0} to {1}".format(valid_set.shape, cleaned_valid_set.shape))

pickle_file_cleaned = 'notMNIST_cleaned.pickle'

try:
    f = open(pickle_file_cleaned, 'wb')
    save = {
        'train_dataset': train_set,
        'train_labels': train_labels,
        'valid_dataset': cleaned_valid_set,
        'valid_labels': cleaned_valid_labels,
        'test_dataset': cleaned_test_set,
        'test_labels': cleaned_test_labels
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file_cleaned)
print('Compressed pickle size:', statinfo.st_size)


'''
Problem 6
Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is
 something to learn, and that it's a problem that is not so trivial that a canned solution solves it. Train a simple
 model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from
 sklearn.linear_model.

 Optional question: train an off-the-shelf model on all the data!
'''

sample_size = 5000
log_reg = LogisticRegression(penalty='l1')
x_train = train_set[:sample_size, :, :].reshape(sample_size, image_size**2)
y_train = train_labels[:sample_size]
log_reg.fit(x_train, y_train)

x_test = test_set.reshape(test_set.shape[0], image_size**2)
y_test = test_labels
print("Accuracy: {}".format(log_reg.score(x_test, y_test)))


#  Sample of some of the estimates

fig = plt.figure()

pred_labels = log_reg.predict(x_test)
pred_probs = log_reg.predict_proba(x_test).round(2)

fig.suptitle("Logistic Regression Predictions", fontsize=16, fontweight='bold')
fig_gs = gs.GridSpec(2, 10)

for i in range(10):
    ax = fig.add_subplot(fig_gs[0, i])
    ax.imshow(x_test[i].reshape(image_size, image_size))
    ax.set_title('{}'.format(chr(y_test[i] + ord('A'))))
    ax.set_axis_off()
    ax2 = fig.add_subplot(fig_gs[1, i])
    ax2.bar(range(len(pred_probs[i])), pred_probs[i])
    if i == 0:
        ax2.set_ylabel("Prediction Probabilities")
    ax2.set_xticks(range(len(pred_probs[i])))
    ax2.set_xticklabels(
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])  # Can use chr(label[i]+ord('A')) trick as well
    ax2.yaxis.set_visible(False)
fig.savefig("plots/Logistic Regression Predictions")

#  Lets look at the features for each class

fig = plt.figure()
fig.suptitle("Features from Logistic Regression", fontsize=16, fontweight='bold')
fig_gs = gs.GridSpec(1, 10)
features = log_reg.coef_
for i in range(10):
    ax = fig.add_subplot(fig_gs[0, i])
    ax.imshow(features[i].reshape(image_size, image_size))
    ax.set_title('{}'.format(chr(i + ord('A'))))
    ax.set_axis_off()
fig.savefig("plots/Features from Logistic Regression")

