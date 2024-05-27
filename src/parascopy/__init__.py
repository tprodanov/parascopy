
__pkg_name__ = 'parascopy'
__title__ = 'Parascopy'
__version__ = '1.16.0-alpha7'
__author__ = 'Timofey Prodanov, Vikas Bansal'
__license__ = 'MIT'

def long_version():
    authors = __author__.split(', ')
    if len(authors) > 1:
        authors_str = ', '.join(authors[:-1]) + ' & ' + authors[-1]
    else:
        authors_str = authors[0]
    return '{} v{}\nCreated by {}'.format(__title__, __version__, authors_str)
