"""
CEPPreprocessing.py

This module is used to do all of the preprocessing for the CEP code.  This
includes downloading records from NCBI (via BioPython), parsing those records
into a local sequence records, etc.

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


import os, re, unittest, time, cPickle
from Bio import Entrez
from pycsa.CEPLogging import LogPipeline

# decorator function to be used for logging purposes
log_function_call = LogPipeline.log_function_call

def text_concatenate(inputList,splicer=' '):
    """Simple helper function to put string descriptions back together"""
    return reduce(lambda x,y: x+splicer+y,inputList)

    
def clean_sequence(inputString):
    """Simple helper function to return a cleaned sequence"""
    return text_concatenate(re.findall('[A-Za-z]+',inputString),splicer='').upper()


class GenbankRetrieve(object):
    """Simple tools to retrieve Genbank records (via Genbank id) of sequences"""
    @log_function_call('Reading Genbank Ids from File')
    def __init__(self,inputFile):
        try: 
            inputFile = open(inputFile,'r')
            handle = inputFile.read()
            inputFile.close()
            genbankIds = re.findall('gi\|(\d+)\|',handle)
        except IOError:
            raise GenbankIOException
        else:
            self.genbankIds = genbankIds

    @log_function_call('Downloading Genebank Ids')
    def download_genbank_records(self,outputFile):
        """Used to retrieve and write Genbank text record from an input list of Genbank ids"""
        print "Number of sequences to retrieve: ", len(self.genbankIds)
        counter = 1
        try:
            outputFile = open(outputFile,'a')
        except IOError:
            raise GenbankIOException
        else:
            for genbankId in self.genbankIds:
                print "Fetching sequence # %d, gi:%s"%(counter,genbankId)
                try:
                    genbankHandle = Entrez.efetch(db="protein", id=genbankId, rettype="gb")
                    genbankRecord = genbankHandle.read()
                    outputFile.write(genbankRecord)
                    counter += 1
                except ValueError:
                    print "Download of gi:%s failed, moving on to next sequence . . ."%(genbankId)
                    print "Will try again later . . ."
                    genbankIds.append(genbankId) # add current ID to end of the list
                except IOError:
                    print "NCBI server is temporarily rejecting jobs . . ."
                    print "Trying again in 15 seconds . . ."
                    time.sleep(15)
                    genbankIds.append(genbankId) # add current ID to end of the list
            outputFile.close()

class GenbankRecord(object):
    """Simple object for a genbank protein record to be read from a string or file name"""
    def __init__(self,recordFile=None):
        if recordFile == None:
            # create an empty Genbank record
            pass
        else:
            # create from a file or a raw string
            if os.path.exists(recordFile):
                # first check for a file to read in 
                recordFile = open(recordFile,'r')
                recordHandle = recordFile.read()
                recordFile.close()
            else:
                # next check for a string raw string
                recordHandle = recordFile
            self.record = recordHandle
            # Parse the record for LOCUS information
            locus = re.search('LOCUS .+',self.record)
            try:
                self.locus = locus.group().split()[1]
                self.length = int(locus.group().split()[2])
            except AttributeError:
                self.locus = None
                self.length = None
            # Parse the record for TITLE information
            try:
                self.title = self.record.split('TITLE')[1].split('JOURNAL')[0].replace('\n','').replace('\t',' ').replace('  ',' ').strip()
            except IndexError:
                self.title = None
            # Parse the record for DEFINITION information
            definition = self.record.split('DEFINITION')[1].split('ACCESSION')[0]
            self.definition = text_concatenate(definition.strip().split())
            # Parse the record for /organism information
            organism = re.search('ORGANISM.+',self.record)
            try:
                # concatenate to get "Genus species"
                self.organism = text_concatenate(organism.group().split()[1:3])
            except AttributeError:
                self.organism = None
            # Parse the record for /strain information
            strain = re.search('/strain="(.+)"',self.record)
            try:
                self.strain = strain.group(1)
            except AttributeError:
                self.strain = None
            # Parse the record for /locus_tag information
            locus_tag = re.search('/locus_tag="(.+)"',self.record)
            try:
                self.locus_tag = locus_tag.group(1)
            except AttributeError:
                self.locus_tag = None
            # Parse the record for genbank id information
            genbankId = re.search('VERSION.+GI:(\d+)',self.record)
            try:
                self.gi = genbankId.group(1)
            except AttributeError:
                self.gi = None
            # Parse the record for /sub_species information
            sub_species = re.search('/sub_species="(.+)"',self.record)
            try:
                self.sub_species = sub_species.group(1)
            except AttributeError:
                self.sub_species = None
            # Parse the record for /sub_strain information
            sub_strain = re.search('/sub_strain="(.+)"',self.record)
            try:
                self.sub_strain = sub_strain.group(1)
            except AttributeError:
                self.sub_strain = None
            # Parse the record for /coded_by information
            coded_by = re.search('/coded_by="(.+)"',self.record)
            try:
                self.coded_by = coded_by.group(1)
            except AttributeError:
                self.coded_by = None
            # Parse the record for sequence information
            sequence = re.search('ORIGIN([\sa-z0-9]+)',self.record)
            try:
                self.sequence = clean_sequence(sequence.group(1))
            except AttributeError:
                self.sequence = None
            # remove the large parsed text
            del self.record
            
    def __repr__(self):
        return "GenbankRecord.{%s}"%(reduce(lambda x,y: x+', '+y,self.__dict__.keys()))
        
class GenbankMultipleRecords(dict):
    """Object that holds multiple Genbank records, initialized from a single Genbank file or string (multiple records)"""
    @log_function_call('Creating Multiple Genbank Records')
    def __init__(self,recordFile=None):
        if recordFile == None:
            # create an empty Genbank records
            pass
        else:
            # create from a file or a raw string
            if os.path.exists(recordFile):
                # first check for a file to read in 
                recordFile = open(recordFile,'r')
                recordHandle = recordFile.read()
                recordFile.close()
            elif type(recordFile) == 'str':
                # next check for a string raw string
                recordHandle = recordFile
            else:
                raise GenbankMultipleRecordsIOException
            for record in [GenbankRecord(x) for x in recordHandle.split('//\n') if 'LOCUS' in x]:
                self[record.gi] = record
        
    @log_function_call('Dumping Genbank Records')
    def dump_genbank_records(self,outputFile):
        """Dump genbank records to a file using the cPickle module"""
        outputFile = open(outputFile,'w')
        cPickle.dump(self,outputFile)
        outputFile.close()
        
class SequenceUtilities(object):
    """These are tools for simple sequence processing"""
    @staticmethod
    def load_genbank_records(inputFile):
        """Load cPickled Genbank records into an empty GenbankMultipleRecords object"""
        newGenbankMultipleRecords = GenbankMultipleRecords()
        inputFile = open(inputFile,'r')
        newGenbankMultipleRecords = cPickle.load(inputFile)
        inputFile.close()
        return newGenbankMultipleRecords

    @staticmethod
    def convert_genbank_fasta(gmrObject):
        """Converts sequences from GenbankMultipleRecords to a fasta.  Return dictionary."""
        fastaDict = {}.fromkeys(gmrObject)
        for record in fastaDict:
            fastaDict[record] = gmrObject[record].sequence
        return fastaDict
    
    @staticmethod
    def read_fasta_sequences(seqFile):
        """Read in sequences from fasta file and return dictionary"""
        seqDict = dict()
        seqFile = open(seqFile,'r')
        listInput = seqFile.read().split('>')[1:]
        seqFile.close()
        for seqRecord in listInput:
            reSeq = re.findall('.+',seqRecord)
            seq = str()
            for seqPart in reSeq[1:]:
                seq += seqPart
            seqDict[reSeq[0].strip()]=seq
        return seqDict
  
    @staticmethod
    def write_fasta_sequences(seqDict,outputFile):
         """Write dictionary sequences to fasta file"""
         outputFile = open(outputFile,'w')
         for seq in seqDict:
             outputFile.write(">%s\n%s\n"%(seq,seqDict[seq]))
         outputFile.close()
 
    @staticmethod
    def read_stockholm_sequences(stoFile):
        """Read in sequences from a stockholm file and return dictionary.
        NOTE: '#=GC RF' the reference sequence that is kept."""
        # load a stockholm file
        stoFile = open(stoFile,'r')
        stoHandle = stoFile.read().split('\n\n')
        stoFile.close()
        # start parsing
        sequences = {}
        for block in stoHandle[1:-1]:
            for line in block.split('\n'):
                parts = line.split()
                # reference sequence for posterior probabilities
                if parts[0] == '#=GR':
                    pass
                # reference sequence for alignment (keep)
                elif parts[0] == '#=GC':
                    try:
                        sequences[parts[0]+' '+parts[1]] += parts[2].replace('.','-')
                    except KeyError:
                        sequences[parts[0]+' '+parts[1]] = parts[2].replace('.','-')
                # add the real sequences
                else:
                    try:
                        sequences[parts[0]] += parts[1].replace('.','-')
                    except KeyError:
                        sequences[parts[0]] = parts[1].replace('.','-')
        return sequences

    @staticmethod
    def write_stockholm_sequences(seqDict,outputFile):
        """Write dictionary sequences to stockholm file"""
        outputFile = open(outputFile,'w')
        outputFile.write("# STOCKHOLM 1.0\n\n")
        for seq in seqDict:
            outputFile.write("%s\t%s\n"%(seq.split(' ')[0],seqDict[seq].replace('-','.')))
        outputFile.write("//\n")
        outputFile.close()
 
    @staticmethod
    @log_function_call('Filtering Sequences by Length')
    def prune_by_length(seqDict,minLen=0,maxLen=10000):
        """Remove sequences in dictionary that exceed length limits and return new dictionary"""
        newDict = dict()
        for seq in seqDict:
            if len(seqDict[seq]) > minLen and len(seqDict[seq]) < maxLen:
                newDict[seq] = seqDict[seq]
        return newDict
        
    @staticmethod
    @log_function_call('Filtering Sequences by Similarity')
    def prune_by_similarity(sequences,similarity=0.95):
        """Remove sequences that are more similar than similarity.  
        Note: removes random sequences."""
        pairwise = {}
        if '#=GC RF' in sequences:
            # stockholm reference file gives HMM aligned positions
            # rank sequences so that you only compute upper triangular matrix
            r = sorted([x for x in sequences.keys() if x != '#=GC RF'])
            for i in xrange(len(r)):
                for j in xrange(i+1,len(r)):
                    pairs = [(x[0],x[1]) for x in zip(sequences[r[i]],sequences[r[j]],sequences['#=GC RF']) if (x[0],x[1]) != ('-','-') and x[2] == 'x']
                    length = float(len(pairs))
                    pairwise[(r[i],r[j])] = len([x for x in pairs if x[0] == x[1]])/length
        else:
            # no stockholm reference to use
            # rank sequences so that you only compute upper triangular matrix
            r = sorted(sequences.keys())
            for i in xrange(len(r)):
                for j in xrange(i+1,len(r)):
                    pairs = [x for x in zip(sequences[r[i]],sequences[r[j]]) if x != ('-','-')]
                    length = float(len(pairs))
                    pairwise[(r[i],r[j])] = len([x for x in pairs if x[0] == x[1]])/length
        while max(pairwise.values()) > similarity:
            for s1,s2 in pairwise:
                if pairwise[(s1,s2)] > similarity:
                    sequences.pop(s1)
                    break
            pairs = [p for p in pairwise.keys() if s1 in p]
            for p in pairs:
                pairwise.pop(p)
        return sequences

    @staticmethod
    @log_function_call('Replacing Nonstandard Amino Acids')
    def replace_nonstandard(sequences,badSymbols=['X','B','Z']):
        """Replaces any of badSymbols in a dictionary of sequences with a gap.  Modifies
        dictionary of sequences in place."""
        for k in sequences:
            listform = list(sequences[k])
            for i in xrange(len(listform)):
                for bad in badSymbols:
                    if listform[i] == bad:
                        listform[i] = '-'
            sequences[k] = ''.join(listform)


class GenbankIOException(IOError):
    @log_function_call('ERROR : IO File')
    def __init__(self):
        print "There is a problem loading your input/output file.  Check the path and file name."

class GenbankMultipleRecordsIOException(IOError):
    @log_function_call('ERROR : Genbank Multiple Records File')
    def __init__(self):
        print "You much initialize a Genbank Multiple Records with a file string or a file handle.  Check your selection."
        
class GenbankRecordTests(unittest.TestCase): 
    """Simple tests to make sure a Genbank record is being parsed correctly"""
    def setUp(self):
        pass
        
if __name__ == '__main__':
    unittest.main()        
        
        
        
        
        
        
