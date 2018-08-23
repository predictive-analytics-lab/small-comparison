from data.objects.sample import Sample
from data.objects.ricci import Ricci
from data.objects.adult import Adult
from data.objects.german import German
from data.objects.propublica_recidivism import PropublicaRecidivism
from data.objects.propublica_violent_recidivism import PropublicaViolentRecidivism
from data.objects.two_gaussians import TwoGaussians
from .flipped_labels import FlippedLabels

DATASETS = [
    TwoGaussians(),
    Ricci(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    PropublicaViolentRecidivism(),
    FlippedLabels(),
]

# For testing, you can just use a sample of the data.  E.g.:
# DATASETS = [ Sample(Adult(), 50) ]
# DATASETS = [Sample(d, 10) for d in DATASETS]

def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names
