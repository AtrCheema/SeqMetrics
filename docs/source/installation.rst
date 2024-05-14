Installation
*************

using pip
=========
The most easy way to install seqmetrics is using ``pip``
::
    pip install SeqMetrics

However, if you are interested in installing all dependencies of seqmetrics, you can
choose to install all of them as well.
::
    pip install SeqMetrics[all]

This will install scipy and easy_mpl libraries. The scipy library is used to calculate some
additional metrics such as kendall_tau while easy_mpl is used for plotting purpose.

We can also specify the seqmetrics version that we want to install as below
::
    pip install SeqMetrics==1.3.2

To updated the installation run
::
    pip install --upgrade SeqMetrics

using github link
=================
You can use github link for install SeqMetrics.
::
    python -m pip install git+https://github.com/AtrCheema/SeqMetrics.git

using setup.py file
===================
go to folder where repository is downloaded
::
    python setup.py install
