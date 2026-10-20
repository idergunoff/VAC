"""Compatibility with trusted legacy .vac pickle files.

Pickle can execute code. Only use load_trusted with files from a trusted source.
These settings require the original measurements to be loaded; they are not an
archive of a study. Keep the pickle protocol and field values unchanged.
"""
import pickle


def load_trusted(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def save(path, options):
    with open(path, 'wb') as stream:
        pickle.dump(options, stream)
