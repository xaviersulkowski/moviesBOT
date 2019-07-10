import io
import os
import zipfile
import requests


def main():
    file_url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    basepath = os.path.dirname(__file__)

    r = requests.get(file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=os.path.join(basepath, '../data'))
    z.close()


if __name__ == '__main__':
    main()
