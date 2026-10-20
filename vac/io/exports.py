"""Legacy output adapters; callers choose paths, defaults match old exports."""


def write_table(table, path, *, sep='\t'):
    table.to_csv(path, sep=sep)


def write_figure(figure, path, **options):
    figure.savefig(path, **options)


def write_document(document, path):
    document.save(path)
