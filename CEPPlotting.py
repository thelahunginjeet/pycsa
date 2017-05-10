"""
CEPPlotting.py

This module is used to make all of the plots for
CA Brown, KS Brown, (2010) PLoS One.  They are all more or
less built up from matplotlib and pylab.

This is a pure python package for correlated substitution analysis.  Specifically, it can be
used for the kinds of analyses in the following publication:

"Validation of coevolving residue algorithms via pipeline sensitivity analysis: ELSC and
OMES and ZNMI, oh my!" PLoS One, e10779. doi:10.1371/journal.pone.0010779 (2010)

@author: Kevin S. Brown (University of Connecticut), Christopher A. Brown (Palomidez LLC)

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2014, Kevin S. Brown and Christopher A. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

3. Neither the name of the University of Connecticut nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys, os, unittest, pylab, math, numpy, copy, networkx, scipy
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter
import networkx as nx


basic_colors = ('b', 'g', 'r', 'c', 'm', 'k','y')
basic_symbols = ('o', 's', '^', 'v', '<', ">", 'x', 'D', 'h', 'p')
basic_lines = ('-', '--', '-.', ':')


def ColorWheel(colors = basic_colors, symbols = basic_symbols,
               lines = basic_lines):
    """
    ColorWheel()

    Returns a generator that cycles through a selection of colors, symbols, and
    line styles for matlibplot.matlab.plot.
    """
    if not colors:
        colors = ('',)
    if not symbols:
        symbols = ('',)
    if not lines:
        lines = ('',)

    while 1:
        for l in lines:
           for s in symbols:
                for c in colors:
                   yield (c, s, l)


class CEPPlotting(object):
    """This is the main plotting class for making scatter histograms, network plots, etc."""
    def __init__(self,figDirectory=None,figFormat='pdf'):
        """Directory to save figures.  Defaults to None, which just does a show()."""
        self.figFormat = figFormat
        self.figDirectory = figDirectory
        self.colorWheel = ColorWheel()
        # create the figure directory if it does not exist and it is not none
        if self.figDirectory is not None and not os.path.exists(self.figDirectory):
            os.mkdir(self.figDirectory)
        # clear the axis limits
        self.accMin = None
        self.accMax = None
        self.repMin = None
        self.repMax = None

    def acc_formatter(self,x,pos):
        return '%0.2f'%x

    def rep_formatter(self,x,pos):
        return '%1.0f'%x

    def set_axis_limits(self,accLimits=(None,None),repLimits=(None,None)):
        """Manual adjustment of accuracy/reproducibility axis limits.  If the function has not been
        called (meaning the limits have not yet been defined), they are automatically calculated."""
        self.accMin = accLimits[0]
        self.accMax = accLimits[1]
        self.repMin = repLimits[0]
        self.repMax = repLimits[1]

    def plot_accuracy_reproducibility(self,*args):
        """Makes an accuracy/reproducibilty plot for a list of CEPPipeline objects.  All objects
        are assumed to have the same resampling scheme and definitions for reproducibility and
        accuracy."""
        # set up colors/points/lines
        self.cepColors = {}.fromkeys(args)
        self.cepPoints = {}.fromkeys(args)
        self.cepLines = {}.fromkeys(args)
        self.inflation = 0.2
        self.kdepoints = 256
        # determine axis limits if they do not exist; if any one of the
        #    limits is none
        if (self.accMin is None) or (self.accMax is None) or (self.repMin is None) or (self.repMax is None):
            self.compute_axis_limits(*args)
        for i in xrange(len(args)):
            (c,s,l) = self.colorWheel.next()
            # self.cepColors[args[i]] = colors[int(math.fmod(i,len(colors)))]
            self.cepColors[args[i]] = c
            self.cepPoints[args[i]] = s
            self.cepLines[args[i]] = l
        # plot and axis locations
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.05
        mainPlot = [left, bottom, width, height]
        accuracyHist = [left, bottom_h, width, 0.2]
        reprodHist = [left_h, bottom, 0.2, height]
        figure1 = pylab.figure(1, figsize=(8,8))
        axMain = pylab.axes(mainPlot,frameon=False)
        if len(args[0].statistics['reproducibility'].values()) > 1:
            # main plot
            self.plot_acc_rep_scatter(axMain,*args)
            # accuracy histogram
            axAccuracy = pylab.axes(accuracyHist,frameon=False)
            axAccuracy.get_yaxis().set_visible(False)
            self.plot_acc_histogram(axAccuracy,*args)
            # reproducibility histogram
            axReprod = pylab.axes(reprodHist,frameon=False)
            axReprod.get_xaxis().set_visible(False)
            self.plot_rep_histogram(axReprod,*args)
        else:
            # reproducibility is a property of the ensemble
            # main plot
            self.plot_acc_rep_line(axMain,*args)
            # accuracy histogram
            axAccuracy = pylab.axes(accuracyHist,frameon=False)
            axAccuracy.get_yaxis().set_visible(False)
            self.plot_acc_histogram(axAccuracy,*args)
            # empty reproducibility histogram, to get y-axis and tick labels
            axReprod = pylab.axes(reprodHist,frameon=False)
            axReprod.get_xaxis().set_visible(False)
            self.plot_rep_histogram(axReprod)
        if self.figDirectory is None:
            pylab.show()
        else:
            pylab.savefig(os.path.join(self.figDirectory,'accuracy_reproducibility')+'.'+self.figFormat,format=self.figFormat,bbox_inches='tight')


    def compute_axis_limits(self, *args):
        """Determines the scales for accuracy and reproducibility; these vary based on the
        particular definition used."""
        self.repMin,self.repMax = 0.0,1.0
        self.accMin,self.accMax = numpy.inf,-numpy.inf
        # figure out the accuracy scale; we can simultaneously set the reproducibilty scale
        for cep in args:
            if min(cep.statistics['reproducibility'].values()) < 0.0:
                self.repMin = -1.0
            self.accMin = min(self.accMin,min(cep.statistics['accuracy'].values()))
            self.accMax = max(self.accMax,max(cep.statistics['accuracy'].values()))
        # push the limits out a bit
        self.accMin = self.accMin - self.inflation*abs(self.accMin)
        self.accMax = self.accMax + self.inflation*abs(self.accMax)


    def plot_acc_rep_scatter(self,scatterAx,*args):
        """Makes a scatterplot of accuracy vs. reproducibility.  Reproducibility is defined for each
        subsample."""
        labelList = list()
        for cep in args:
            accVals = cep.statistics['accuracy'].values()
            repVals = cep.statistics['reproducibility'].values()
            scatterAx.plot(accVals,repVals,self.cepColors[cep]+self.cepPoints[cep],mec=self.cepColors[cep],alpha=0.5)
            labelList.append(cep.fileIndicator+' '+str(cep.numSequences)+' '+cep.method)
        scatterAx.set_xlim((self.accMin,self.accMax))
        scatterAx.set_ylim((self.repMin,self.repMax))
        scatterAx.get_xaxis().set_visible(False)
        scatterAx.get_yaxis().set_visible(False)
        pylab.legend(tuple(labelList),loc='best')


    def plot_acc_rep_line(self,lineAx,*args):
        """Similar to plot_acc_rep_scatter, except each pipeline has only one reproducibility value.  The figure consequently
        looks different."""
        labelList = list()
        lineList = list()
        for cep in args:
            accVals= cep.statistics['accuracy'].values()
            repVal = cep.statistics['reproducibility'].values()[0]
            lineAx.plot([min(accVals),max(accVals)],[repVal,repVal],self.cepColors[cep]+self.cepLines[cep],alpha=0.5,lw=3)
            l = lineAx.plot([numpy.mean(accVals)],[repVal],self.cepPoints[cep],color=self.cepColors[cep],mfc=self.cepColors[cep],mec=self.cepColors[cep],markersize=12)
            lineList.append(l[0])
            labelList.append(cep.fileIndicator+' '+str(cep.numSequences)+' '+cep.method)
        # vertical line at right
        lineAx.set_xlim((self.accMin,self.accMax))
        lineAx.set_ylim((self.repMin,self.repMax))
        lineAx.get_xaxis().set_visible(False)
        lineAx.get_yaxis().set_visible(False)
        pylab.legend(lineList,labelList,loc='best')


    def plot_acc_histogram(self,histAx,*args):
        """Makes a plot of the accuracy histogram (normal orientation)"""
        maxAccHeight = 0.01
        for cep in args:
            kde = gaussian_kde(cep.statistics['accuracy'].values())
            support = numpy.linspace(self.accMin,self.accMax,self.kdepoints)
            mPDF = kde(support)
            histAx.plot(support,mPDF,self.cepColors[cep],lw=3)
            # readjust height
            if max(mPDF) > maxAccHeight:
                maxAccHeight = max(mPDF)
        # add axis line and clean up
        formatter = FuncFormatter(self.acc_formatter)
        histAx.plot([self.accMin,self.accMax],[0,0],'k-',lw=3)
        histAx.set_xlim((self.accMin,self.accMax))
        histAx.set_ylim((0,1.1*maxAccHeight))
        histAx.set_xticks([self.accMin,self.accMax])
        histAx.tick_params(axis='x',direction='in',top=False)
        histAx.xaxis.set_major_formatter(formatter)
        histAx.set_yticks([])


    def plot_rep_histogram(self,histAx,*args):
        """Plots the reproducibility histogram (sideways)"""
        maxRepHeight = 0.01
        for cep in args:
            kde = gaussian_kde(cep.statistics['reproducibility'].values())
            support = numpy.linspace(self.repMin,self.repMax,self.kdepoints)
            mPDF = kde(support)
            histAx.plot(mPDF,support,self.cepColors[cep],lw=3)
            # readjust height
            if max(mPDF) > maxRepHeight:
                maxRepHeight = max(mPDF)
        # add axis line and clean up
        formatter = FuncFormatter(self.rep_formatter)
        histAx.plot([0,0],[self.repMin,self.repMax],'k-',lw=3)
        histAx.set_ylim((self.repMin,self.repMax))
        histAx.set_xlim((0,1.1*maxRepHeight))
        histAx.set_yticks([self.repMin,self.repMax])
        histAx.tick_params(axis='y',direction='in',right=False)
        histAx.yaxis.set_major_formatter(formatter)
        histAx.set_xticks([])


    def ring_plot(self,cep,rcut,ncData=None,layout='twopi'):
        """
        Draws a consensus pipeline reproducibility graph and saves it or
        displays it. The nodes can be positioned on a ring or a series of
        concentric circles.  In this representation, labels are suppressed.

        Notice that unlike the histogram plotting functions, this function
        can only plot one pipeline (one consensus network).

        INPUT
        ------
        cep     : CEPPipeline object, required

        rcut    : float in [0,1], required
            Reproducibility threshold for suppressing edges

        ncData  : dict, optional
            custom color information to decorate nodes; should be a dictionary
            keyed on nodes, with

        layout  : string, optional
            suggested layouts are 'circo','twopi',and 'radial'
        """
        # figure out which edges to plot
        edges_to_show = []
        for e in cep.consensusGraph.edges():
            if cep.consensusGraph.get_edge_data(e[0],e[1])['weight'] > rcut:
                edges_to_show.append(e)
        # construct node colors
        nList = cep.consensusGraph.nodes()
        nc = ['k' for x in nList]
        if ncData is not None:
            for i in xrange(0,len(nList)):
                if ncData.has_key(nList[i]):
                    nc[i] = ncData[nList[i]]
        # now make the plot
        pylab.figure(figsize=(8,8))
        if layout == 'radial':
            pos = nx.circular_layout(cep.consensusGraph)
        else:
            pos = nx.nx_pydot.graphviz_layout(cep.consensusGraph,prog=layout)
        nx.draw_networkx(cep.consensusGraph,pos=pos,edgelist=edges_to_show,node_color=nc,node_size=25,with_labels=False)
        pylab.axis('off')
        if self.figDirectory is None:
            pylab.show()
        else:
            pylab.savefig(os.path.join(self.figDirectory,'network_ring_'+cep.method+'_'+str(rcut).replace('.','p'))+'.'+self.figFormat,
                format=self.figFormat,bbox_inches='tight')


    def net_plot(self,cep,rcut,ncData=None,layout='neato',cmap=pylab.cm.RdYlGn):
        """
        Draws a pipeline reproducibility graph and saves it or displays it.
        This version uses energy- or force-based layouts, includes node
        labels, and allows optional color coding as in ring_plot.

        Only supported color coding is floats.

        INPUT
        ------
        cep     : CEPPipeline object, required

        rcut    : float in [0,1], required
            Reproducibility threshold for pruning the pipeline graph

        ncData  : dict, optional
            custom color information to decorate nodes (needs to be floats!).  The
            default nodecolor is 0.0, the color of which will depend on cmap.

        layout  : string, optional
            can be any graphviz layout, but 'neato' or 'fdp' tend to work well

        cmap    : matplotlib.cmap, optional
            colormap for node display; 'RdYlGn' is the default in the hopes that
            node labels are still readable anywhere.
        """
        # copy the graph and prune it
        G = copy.deepcopy(cep.consensusGraph)
        G.prune_graph(rcut)
        nList = G.nodes()
        nc = [0.0 for x in nList]
        if ncData is not None:
            for i in xrange(0,len(nList)):
                if ncData.has_key(nList[i]):
                    nc[i] = ncData[nList[i]]
        # now make the plot
        pylab.figure(figsize=(8,8))
        pos = nx.nx_pydot.graphviz_layout(G,prog=layout)
        nx.draw_networkx(G,pos=pos,node_color=nc,cmap=cmap,node_size=200,with_labels=True,
            font_size=6,width=2.0,alpha=0.5)
        pylab.axis('off')
        if self.figDirectory is None:
            pylab.show()
        else:
            pylab.savefig(os.path.join(self.figDirectory,'network_spring_'+cep.method+'_'+str(rcut).replace('.','p'))+'.'+self.figFormat,
                format=self.figFormat,bbox_inches='tight')


    '''
    # FROM HERE ON OUT THESE MAY NOT WORK - NEED TO BE UPDATED
    def plot_accuracy_versus_cutoff(self, *args):
        """Makes a plot of accuracy versus cutoff for edge resampling
        Note: it take a variable number of arguments (i.e. pass a list of CEPPipeline objects)"""
        # setup colors
        colors = 'rgbymck'
        point = ','
        cepColors = {}.fromkeys(args)
        for i in xrange(len(args)):
            cepColors[args[i]] = colors[int(math.fmod(i,len(colors)))]
        cutoffs = numpy.arange(0,1,0.005).tolist()
        for cep in args:
            distances = list()
            graph = copy.deepcopy(cep.consensusGraph)
            for cutoff in cutoffs:
                graph.prune_graph(cutoff)
                edges = [cep.distances[x] for x in graph.edges()]
                distances.append((scipy.mean(edges),scipy.std(edges)))
            means = [x[0] for x in distances]
            stddevs = [x[1] for x in distances]
            pylab.plot(cutoffs,means,cepColors[cep],lw=3)
            for i in xrange(0,len(stddevs),5):
                pylab.plot([cutoffs[i],cutoffs[i]],[means[i]-stddevs[i],means[i]+stddevs[i]],cepColors[cep],lw=1)
        if self.figDirectory is None:
            pylab.show()
        else:
            pylab.savefig(os.path.join(self.figDirectory,'accuracy_cutoff')+'.'+self.figFormat,format=self.figFormat)


    def plot_accuracy_versus_percolation(self, *args):
        """Makes a plot of accuracy versus percolation (largest connected component) for edge resampling
        Note: it take a variable number of arguments (i.e. pass a list of CEPPipeline objects)"""
        # setup colors
        colors = 'rgbymck'
        point = ','
        cepColors = {}.fromkeys(args)
        for i in xrange(len(args)):
            cepColors[args[i]] = colors[int(math.fmod(i,len(colors)))]
        cutoffs = numpy.arange(0,1,0.005).tolist()
        for cep in args:
            percolation = list()
            graph = copy.deepcopy(cep.consensusGraph)
            for cutoff in cutoffs:
                graph.prune_graph(cutoff)
                components = map(lambda x: len(x), networkx.connected_components(graph))
                if len(components) == 0:
                    percolation.append(0)
                else:
                    percolation.append(max(components))
            pylab.plot(cutoffs,percolation,cepColors[cep],lw=3)
        if self.figDirectory is None:
            pylab.show()
        else:
            pylab.savefig(os.path.join(self.figDirectory,'accuracy_percolation')+'.'+self.figFormat,format=self.figFormat)
    '''

class CEPPlottingTests(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
