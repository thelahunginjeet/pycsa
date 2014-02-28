"""
CEPLogging.py

Created by CA Brown and KS Brown.  For reference see:
Brown, C.A., Brown, K.S. 2010. Validation of coevolving residue algorithms via 
pipeline sensitivity analysis: ELSC and OMES and ZNMI, oh my! PLoS One, e10779. 
doi:10.1371/journal.pone.0010779

This module is used to provide logging functions for all of the CEP code, 
so that a user doesn't have to keep track of what types of nonparametric
decisions / errors were made during a pipeline project.
"""

from time import ctime
import re, os, unittest
from functools import wraps

def clean_string(inputString):
	parts = re.search("'(\w+\.\w+)'",inputString)
	return parts.group(1)
	
def create_footer(lkwargs):
	footer = ""
	for kw in lkwargs:
		footer += "\t%s = %s\n"%(kw, lkwargs[kw])
	return footer

class LogPipeline(object):
	"Simple object to use for logging"
	@classmethod
	def __init__(cls,logfile):
		cls.logfile = logfile
		fileObject = open(cls.logfile,'a')
		timeStamp = "\n[%s]  Initializing Log\n"%(ctime())
		fileObject.write(timeStamp)
		fileObject.close()
		# set last event
		cls.lastEvent = timeStamp

	@classmethod
	def log_function_call(cls,event):
		"""Main function to log function calls
		Note: takes one event and multiple args and kwargs."""
		def wrap(f):
			@wraps(f)
			def decorator(*args, **kwargs):
				if hasattr(cls,'logfile'):
					if cls.lastEvent == event:
						# just called this function; used to thin out the log file
						pass
					else:
						cls.lastEvent = event
						fileObject = open(cls.logfile,'a')
						header = "[%s]  %s\n"%(ctime(),event)
						fileObject.write(header)
						if len(kwargs) > 0:
							footer = "%s"%(create_footer(kwargs))
							fileObject.write(footer)
						fileObject.close()
				else:
					pass
				return f(*args, **kwargs)
			return decorator
		return wrap


class LogIOException(IOError):
	def __init__(self):
		print "There is a problem with your log files.  Check the path and file name."

if __name__ == '__main__':
	pass







