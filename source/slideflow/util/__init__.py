import sys
import json
import csv
import time
import os
import shutil
import datetime

import tensorflow as tf

from glob import glob
from tensorflow.keras import backend as K
from os.path import join, isdir, exists

# Enable color sequences on Windows
try:
	import ctypes
	kernel32 = ctypes.windll.kernel32
	kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except:
	pass
# ------

SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs', 'tiff', 'svslide', 'bif', 'jpg']
SLIDE_ANNOTATIONS_TO_IGNORE = ['', 'na', 'n/a', 'none', 'missing']

HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
PURPLE = '\033[38;5;5m'
UNDERLINE = '\033[4m'

FORMATTING_OPTIONS = [HEADER, BLUE, GREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE]
LOGGING_PREFIXES = ['', ' + ', '    - ']
LOGGING_PREFIXES_WARN = ['', ' ! ', '    ! ']
LOGGING_PREFIXES_EMPTY = ['', '   ', '     ']

# Old BatchNorm fix for bug in TF v1.14
class UpdatedBatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, inputs, training=None):
		true_phase = int(K.get_session().run(K.learning_phase()))
		trainable = int(self.trainable)
		with K.learning_phase_scope(trainable * true_phase):
			return super(tf.keras.layers.BatchNormalization, self).call(inputs, training)

class Bar:
	starttime = None
	lastupdated = None
	text = ''

	def __init__(self, ending_value, starting_value=0, bar_length=20, label='',
					show_eta=False, show_counter=False, counter_text=''):
		self.value = starting_value
		self.end_value = ending_value
		self.bar_length = bar_length
		self.label = label
		self.show_counter = show_counter
		self.counter_text = '' if not counter_text else " " + counter_text
		self.show_eta = show_eta		

	def get_text(self):
		current_time = int(time.time())
		if not self.starttime:
			self.starttime = current_time
			self.lastupdated = self.starttime
		elif current_time == self.lastupdated:
			return self.text
		else:
			self.lastupdated = current_time

		percent = float(self.value) / self.end_value
		arrow = chr(0x2588) * int(round(percent * self.bar_length))
		spaces = u'-' * (self.bar_length - len(arrow))
		timediff = int(time.time())-self.starttime
		if timediff != 0:
			num_per_sec = self.value/timediff

		self.text = u"\u007c{0}\u007c {1:.1f}%{2}".format(arrow + spaces, 
													 (float(self.value) / self.end_value)*100, 
													 f' ({self.label})' if self.label else '')
		if self.show_counter:
			num_per_sec_str = "?" if timediff == 0 else f'{num_per_sec:.1f}'
			self.text += f" {num_per_sec_str}{self.counter_text}/sec"
		if self.show_eta and timediff:
			eta_sec = (self.end_value - self.value) / num_per_sec
			self.text += f" (ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_sec))})"
		elif self.show_eta:
			self.text += f" (ETA: ?)"

		return self.text

class ProgressBar:
	tail = ''
	text = ''

	def __init__(self, ending_val, starting_val=0, bar_length=20, endtext='', show_eta=False, 
					show_counter=False, counter_text='', leadtext=''):
		
		self.leadtext = leadtext
		self.BARS = [Bar(ending_val, starting_val, bar_length, endtext, show_eta, show_counter, counter_text)]
		self.refresh()

	def add_bar(self, val, endval, bar_length=20, endtext='', show_eta=False,
					show_counter=False, counter_text=''):

		self.BARS += [Bar(val, endval, bar_length, endtext, show_eta, show_counter, counter_text)]
		self.refresh()
		return len(self.BARS)-1

	def increase_bar_value(self, id=0):
		self.BARS[id].value = min(self.BARS[id].value + 1, self.BARS[id].end_value)
		self.refresh()

	def set_bar_value(self, value, id=0):
		self.BARS[id].value = min(val, self.BARS[id].end_value)
		self.refresh()

	def set_bar_text(self, text, id=0):
		self.BARS[id].text = text
		self.refresh()

	def refresh(self):
		if len(self.BARS) == 0:
			sys.stdout.write("\r\033[K")
			sys.stdout.flush()
			return
		new_text = f"\r\033[K{self.leadtext}"
		for bar in self.BARS:
			new_text += bar.get_text()
			if len(self.BARS) > 1:
				new_text += "  "
		new_text += self.tail
		if new_text != self.text:
			sys.stdout.write(new_text)
			sys.stdout.flush()
			self.text = new_text

	def end(self, id=-1):
		if id == -1:
			bars_keys = list(self.BARS.keys())
			for bar_id in bars_keys:
				del(self.BARS[bar_id])
			sys.stdout.write(self.text)
		else:
			del(self.BARS[id])
			sys.stdout.write(f"\r\033[K{self.text}\n")

	def print(self, string):
		sys.stdout.write(f"\r\033[K{string}\n")
		sys.stdout.flush()
		sys.stdout.write(self.text)
		sys.stdout.flush()

def warn(text):
	return WARNING + str(text) + ENDC

def header(text):
	return HEADER + str(text) + ENDC

def info(text):
	return BLUE + str(text) + ENDC

def green(text):
	return GREEN + str(text) + ENDC

def fail(text):
	return FAIL + str(text) + ENDC

def bold(text):
	return BOLD + str(text) + ENDC

def underline(text):
	return UNDERLINE + str(text) + ENDC

def purple(text):
	return PURPLE + str(text) + ENDC

class Logger:
	logfile = None
	INFO_LEVEL = 3
	WARN_LEVEL = 3
	ERROR_LEVEL = 3
	COMPLETE_LEVEL = 3
	SILENT = False
	WRITE = False

	def __init__(self):
		pass
	def info(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{info('INFO')}] {text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message
	def warn(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{warn('WARN')}] {text}"
		if print_func and l <= self.WARN_LEVEL:
			print_func(message)
		self.log(message)
		return message
	def error(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES_WARN[l]}[{fail('ERROR')}] {text}"
		if print_func and l <= self.ERROR_LEVEL:
			print_func(message)
		self.log(message)
		return message
	def complete(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{header('Complete')}] {text}"
		if print_func and l <= self.COMPLETE_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message
	def label(self, label, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}[{green(label)}] {text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message
	def empty(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"{LOGGING_PREFIXES[l]}{text}"
		if print_func and l <= self.INFO_LEVEL and not self.SILENT:
			print_func(message)
		self.log(message)
		return message
	def header(self, text, l=0, print_func=print):
		l = min(l, len(LOGGING_PREFIXES)-1)
		message = f"\n{LOGGING_PREFIXES_EMPTY[l]}{bold(text)}"
		if print_func and not self.SILENT:
			print_func(message)
		self.log(message)
		return message
	def log(self, text):
		st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
		if self.logfile and self.WRITE:
			for s in FORMATTING_OPTIONS:
				text = text.replace(s, "")
			outfile = open(self.logfile, 'a')
			outfile.write(f"[{st}] {text.strip()}\n")
			outfile.close()

log = Logger()

class TCGA:
	patient = 'submitter_id'
	project = 'project_id'
	slide = 'slide'

def make_dir(_dir):
	'''Makes a directory if one does not already exist, in a manner compatible with multithreading. '''
	if not exists(_dir):
		try:
			os.makedirs(_dir, exist_ok=True)
		except FileExistsError:
			pass

def global_path(root, path_string):
	'''Returns global path from a local path.'''
	if not root: root = ""
	if path_string and (len(path_string) > 2) and path_string[:2] == "./":
		return os.path.join(root, path_string[2:])
	elif path_string and (path_string[0] != "/"):
		return os.path.join(root, path_string)
	else:
		return path_string		

def _shortname(string):
	if len(string) == 60:
		# May be TCGA slide with long name; convert to patient name by returning first 12 characters
		return string[:12]
	else:
		return string

def yes_no_input(prompt, default='no'):
	'''Prompts user for yes/no input.'''
	yes = ['yes','y']
	no = ['no', 'n']
	while True:
		response = input(prompt)
		if not response and default:
			return True if default in yes else False
		if response.lower() in yes:
			return True
		if response.lower() in no:
			return False
		print(f"Invalid response.")

def dir_input(prompt, root, default=None, create_on_invalid=False, absolute=False):
	'''Prompts user for directory input.'''
	while True:
		if not absolute:
			response = global_path(root, input(f"{prompt}"))
		else:
			response = input(f"{prompt}")
		if not response and default:
			response = global_path(root, default)
		if not os.path.exists(response) and create_on_invalid:
			if yes_no_input(f'Directory "{response}" does not exist. Create directory? [Y/n] ', default='yes'):
				os.makedirs(response)
				return response
			else:
				continue
		elif not os.path.exists(response):
			print(f'Unable to locate directory "{response}"')
			continue
		return response

def file_input(prompt, root, default=None, filetype=None, verify=True):
	'''Prompts user for file input.'''
	while True:
		response = global_path(root, input(f"{prompt}"))
		if not response and default:
			response = global_path(root, default)
		if verify and not os.path.exists(response):
			print(f'Unable to locate file "{response}"')
			continue
		extension = path_to_ext(response)
		if filetype and (extension != filetype):
			print(f'Incorrect filetype; provided file of type "{extension}", need type "{filetype}"')
			continue
		return response

def int_input(prompt, default=None):
	'''Prompts user for int input.'''
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		try:
			int_response = int(response)
		except ValueError:
			print("Please supply a valid number.")
			continue
		return int_response

def float_input(prompt, default=None, valid_range=None):
	'''Prompts user for float input.'''
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		try:
			float_response = float(response)
		except ValueError:
			print("Please supply a valid number.")
			continue
		if valid_range and not (float_response >= valid_range[0] and float_response <= valid_range[1]):
			print(f"Please supply a valid numer in the range {valid_range[0]} to {valid_range[1]}")
		return float_response

def choice_input(prompt, valid_choices, default=None, multi_choice=False, input_type=str):
	'''Prompts user for multi-choice input.'''
	while True:
		response = input(f"{prompt}")
		if not response and default:
			return default
		if not multi_choice and response not in valid_choices:
			print("Invalid option.")
			continue
		elif multi_choice:
			try:
				response = [input_type(r) for r in response.replace(" ", "").split(',')]
			except:
				print(f"Invalid selection (response: {response})")
				continue
			invalid = [r not in valid_choices for r in response]
			if any(invalid):
				print(f'Invalid selection (response: {response})')
				continue
		return response

def load_json(filename):
	'''Reads JSON data from file.'''
	with open(filename, 'r') as data_file:
		return json.load(data_file)

def write_json(data, filename):
	'''Writes data to JSON file.'''
	with open(filename, "w") as data_file:
		json.dump(data, data_file, indent=1)

def _parse_function(example_proto):
	feature_description = {'slide':     tf.io.FixedLenFeature([], tf.string),
						   'image_raw':	tf.io.FixedLenFeature([], tf.string)}
	return tf.io.parse_single_example(example_proto, feature_description)

def get_slide_paths(slides_dir):
	'''Get all slide paths from a given directory containing slides.'''
	slide_list = [i for i in glob(join(slides_dir, '**/*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS]
	slide_list.extend([i for i in glob(join(slides_dir, '*.*')) if path_to_ext(i).lower() in SUPPORTED_FORMATS])
	return slide_list

def read_annotations(annotations_file):
	'''Read an annotations file.'''
	results = []
	# Open annotations file and read header
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		# First, try to open file
		try:
			header = next(csv_reader, None)
		except OSError:
			log.error(f"Unable to open annotations file {green(annotations_file)}, is it open in another program?")
			sys.exit()

		for row in csv_reader:
			row_dict = {}
			for i, key in enumerate(header):
				row_dict[key] = row[i]
			results += [row_dict]
	return header, results
			
def get_relative_tfrecord_paths(root, directory=""):
	tfrecords = [join(directory, f) for f in os.listdir(join(root, directory)) if (not isdir(join(root, directory, f)) and len(f) > 10 and f[-10:] == ".tfrecords")]
	subdirs = [f for f in os.listdir(join(root, directory)) if isdir(join(root, directory, f))]
	for sub in subdirs:
		tfrecords += get_relative_tfrecord_paths(root, join(directory, sub))
	return tfrecords

def contains_nested_subdirs(directory):
	subdirs = [_dir for _dir in os.listdir(directory) if isdir(join(directory, _dir))]
	for subdir in subdirs:
		contents = os.listdir(join(directory, subdir))
		for c in contents:
			if isdir(join(directory, subdir, c)):
				return True
	return False

def path_to_name(path):
	'''Returns name of a file, without extension, from a given full path string.'''
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return _file
	else:
		return '.'.join(_file.split('.')[:-1])

def path_to_ext(path):
	'''Returns extension of a file path string.'''
	_file = path.split('/')[-1]
	if len(_file.split('.')) == 1:
		return ''
	else:
		return _file.split('.')[-1]

def update_results_log(results_log_path, model_name, results_dict):
	'''Dynamically update results_log when recording training metrics.'''
	# First, read current results log into a dictionary
	results_log = {}
	if exists(results_log_path):
		with open(results_log_path, "r") as results_file:
			reader = csv.reader(results_file)
			headers = next(reader)
			try:
				model_name_i = headers.index('model_name')
				result_keys = [k for k in headers if k != 'model_name']
			except ValueError:
				model_name_i = headers.index('epoch')
				result_keys = [k for k in headers if k != 'epoch']
			for row in reader:
				name = row[model_name_i]
				results_log[name] = {}
				for result_key in result_keys:
					result = row[headers.index(result_key)]
					results_log[name][result_key] = result
		# Move the current log file into a temporary file
		shutil.move(results_log_path, f"{results_log_path}.temp")

	# Next, update the results log with the new results data
	for epoch in results_dict:
		results_log.update({f'{model_name}-{epoch}': results_dict[epoch]})

	# Finally, create a new log file incorporating the new data
	with open(results_log_path, "w") as results_file:
		writer = csv.writer(results_file)
		result_keys = []
		# Search through results to find all results keys
		for model in results_log:
			result_keys += list(results_log[model].keys())
		# Remove duplicate result keys
		result_keys = list(set(result_keys))
		result_keys.sort()
		# Write header labels
		writer.writerow(['model_name'] + result_keys)
		# Iterate through model results and record
		for model in results_log:
			row = [model]
			# Include all saved metrics
			for result_key in result_keys:
				if result_key in results_log[model]:
					row += [results_log[model][result_key]]
				else:
					row += [""]
			writer.writerow(row)

	# Delete the old results log file
	if exists(f"{results_log_path}.temp"):
		os.remove(f"{results_log_path}.temp")