"""
run_pipeline.py

Created by CA Brown and KS Brown.  For reference see:
Brown, C.A., Brown, K.S. 2010. Validation of coevolving residue algorithms via
pipeline sensitivity analysis: ELSC and OMES and ZNMI, oh my! PLoS One, e10779.
doi:10.1371/journal.pone.0010779

This is the main test script for the pipeline.  All of the files that are needed
are included in the '../tests' directory.  The datasets is small enough (subset of the PDZ
dataset from our publication) to run a small number of resamplings in a short amount of time.

The suggested way to run this is to install pycsa via:

pip install pycsa/ --upgrade

then copy the test directory to another location on your system, and run the
test script using

python run_test.py

This version calculates resampling statistics and produces a scatterplot of
the accuracy/reproducibility results.
"""

from pycsa import CEPPipeline,CEPPlotting,CEPLogging
import sys,os


def run_pipelines(method_list,options):
    # create logfile
    #   note: if the logfile already exists, new time-stamped logging statements will be appended to this file
    logger = CEPLogging.LogPipeline('./pdzlog.txt')

    # create a new pipeline using input options
    pipe = CEPPipeline.CEPPipeline(options=options)

    # issue this command to clean the network and alignment directories
    #   note: the pipeline won't overwrite network & alignment files (to avoid unnecessary recalculation) so you have to issue
    #          clean project or manually by delete the files
    pipe.clean_project()

    # first, make subalignments by resampling
    #   note: this will create a directory in main_directory/ named 'alignments'
    pipe.resample_alignment()

    # calculate the networks using the specified methods
    #   note: this creates a directory in the main_directory/ named 'networks'
    pipe.calculate_networks(method_list=method_list)

    # calculate graphs, resampling stats, and write a results database for each method
    #   note: the database also contains a copy of the options structure used for the calculations
    for m in method_list:
        pipe.read_graphs(method=m)
        # these are independent; any one can be called without calling the other two
        pipe.calculate_accuracy()
        pipe.calculate_reproducibility()
        pipe.calculate_consensus_graph()
        # write the result
        #   note: this creates a directory in the main_directory/ named 'databases'
        pipe.write_database()
    return


def plot_pipelines(method_list,main_directory,options):
    # read in the databases written by run_pipeline
    plot = CEPPlotting.CEPPlotting(figDirectory=main_directory+options.fig_directory)
    plot.set_axis_limits(accLimits=(0,1),repLimits=(-1,1))
    ceps = {}.fromkeys(method_list)
    for meth in method_list:
        db_file = main_directory+'databases/pdz_800_'+meth+'.pydb'
        # just pass None to options, as they will be read from the database file
        ceps[meth] = CEPPipeline.CEPPipeline(options=None,database=db_file)
        plot.net_plot(ceps[meth],0.75)
    plot.plot_accuracy_reproducibility(*ceps.values())



if __name__ == '__main__':
    # absolutely required to do a run
    main_directory = './'                                  # base project directory
    alignment_file = 'pdz_test.aln'                        # master alignment file
    canon_sequence = '1IU0'

    # create the options structure; most parameters have default values
    options = CEPPipeline.CEPParameters(main_directory,alignment_file,canon_sequence)

    # modify some of the options (see CEPPipeline for a full list of options and their defaults)
    options.set_parameters(pdb_file='1iu0.pdb',num_partitions=10,file_indicator='pdz',resampling_method='splithalf',acc_method='bacc')

    # list of methods (see CEPAlgorithms for the full list of methods)
    method_list = ['mi','zres','mip']

    # run
    run_pipelines(method_list,options)

    # plot results
    plot_pipelines(method_list,main_directory,options)
