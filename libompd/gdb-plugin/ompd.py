import ompdModule
import gdb
import re
import traceback
from ompd_address_space import ompd_address_space
from frame_filter import FrameFilter
from enum import Enum


addr_space = None
ff = None
icv_map = None
ompd_scope_map = {1:'global', 2:'address_space', 3:'thread', 4:'parallel', 5:'implicit_task', 6:'task'}
in_task_function = False

class ompd(gdb.Command):
	def __init__(self):
		super(ompd, self).__init__('ompd',
			gdb.COMMAND_STATUS,
			gdb.COMPLETE_NONE,
			True)

class ompd_init(gdb.Command):
	"""Find and initialize ompd library"""

	# first parameter is command-line input, second parameter is gdb-specific data
	def __init__(self):
		self.__doc__ = 'Find and initialize OMPD library\nusage: ompd init'
		super(ompd_init, self).__init__('ompd init',
						gdb.COMMAND_DATA)

	def invoke(self, arg, from_tty):
		global addr_space
		global ff
		try:
			try:
				print(gdb.newest_frame())
			except:
				gdb.execute("start")
			try:
				lib_list = gdb.parse_and_eval("(char**)ompd_dll_locations")
			except gdb.error:
				raise ValueError("No ompd_dll_locations symbol in execution, make sure to have an OMPD enabled OpenMP runtime");
			
			while(gdb.parse_and_eval("(char**)ompd_dll_locations") == False):
				gdb.execute("tbreak ompd_dll_locations_valid")
				gdb.execute("continue")
			
			lib_list = gdb.parse_and_eval("(char**)ompd_dll_locations")
			
			i = 0
			while(lib_list[i]):
				ret = ompdModule.ompd_open(lib_list[i].string())
				if ret == -1:
					raise ValueError("Handle of OMPD library is not a valid string!")
				if ret == -2:
					print("ret == -2")
					pass # It's ok to fail on dlopen
				if ret == -3:
					print("ret == -3")
					pass # It's ok to fail on dlsym
				if ret < -10:
					raise ValueError("OMPD error code %i!" % (-10 - ret))
					
				if ret > 0:
					print("Loaded OMPD lib successfully!")
					try:
						addr_space = ompd_address_space()
						ff = FrameFilter(addr_space)
					except:
						traceback.print_exc()
					return
				i = i+1
			
			raise ValueError("OMPD library could not be loaded!")
		except:
			traceback.print_exc()

class ompd_threads(gdb.Command):
	"""Register thread ids of current context"""
	def __init__(self):
		self.__doc__ = 'Register information on threads of current context\n usage: ompd threads'
		super(ompd_threads, self).__init__('ompd threads',
						gdb.COMMAND_STATUS)
	
	def invoke(self, arg, from_tty):
		global addr_space
		addr_space.list_threads(True)

def curr_thread():
	"""Helper function for ompd_step. Returns the thread object for the current thread number."""
	global addr_space
	return addr_space.threads[int(gdb.selected_thread().num)]

class ompd_test(gdb.Command):
	"""Test area"""
	def __init__(self):
		self.__doc__ = 'Test functionalities for correctness\nusage: ompd test'
		super(ompd_test, self).__init__('ompd test',
						gdb.COMMAND_OBSCURE)
	
	def invoke(self, arg, from_tty):
		global addr_space
		
		# get task function for current task of current thread
		current_thread = int(gdb.selected_thread().num)
		current_thread_obj = addr_space.threads[current_thread]
		task_function = current_thread_obj.get_current_task().get_task_function()
		print("bt value:", int("0x0000000000400b6c",0))
		print("get_task_function value:", task_function)
		
		# get task function of implicit task in current parallel region for current thread
		current_parallel_obj = current_thread_obj.get_current_parallel()
		task_in_parallel = current_parallel_obj.get_task_in_parallel(current_thread)
		task_function_in_parallel = task_in_parallel.get_task_function()
		print("task_function_in_parallel:", task_function_in_parallel)

class ompd_bt(gdb.Command):
	"""Turn filter for 'bt' on/off for output to only contain frames relevant to the application or all frames."""
	def __init__(self):
		self.__doc__ = 'Turn filter for "bt" output on or off. Specify "continued" option to trace worker threads back to master threads.\nusage: ompd bt on|on continued|off'
		super(ompd_bt, self).__init__('ompd bt',
					gdb.COMMAND_STACK)
	
	def invoke(self, arg, from_tty):
		global ff
		global addr_space
		global icv_map
		global ompd_scope_map
		if icv_map is None:
			icv_map = {}
			current = 0
			more = 1
			while more > 0:
				tup = ompdModule.call_ompd_enumerate_icvs(addr_space.addr_space, current)
				(current, next_icv, next_scope, more) = tup
				icv_map[next_icv] = (current, next_scope, ompd_scope_map[next_scope])
			print('Initialized ICV map successfully for filtering "bt".')
		
		arg_list = gdb.string_to_argv(arg)
		if len(arg_list) == 0:
			print('When calling "ompd bt", you must either specify "on", "on continued" or "off". Check "help ompd".')
		elif len(arg_list) == 1 and arg_list[0] == 'on':
			addr_space.list_threads(False)
			ff.set_switch(True)
			ff.set_switch_continue(False)
		elif arg_list[0] == 'on' and arg_list[1] == 'continued':
			ff.set_switch(True)
			ff.set_switch_continue(True)
		elif arg_list[0] == 'off':
			ff.set_switch(False)
			ff.set_switch_continue(False)
		else:
			print('When calling "ompd bt", you must either specify "on", "on continued" or "off". Check "help ompd".')

# TODO: remove
class ompd_taskframes(gdb.Command):
	"""Prints task handles for relevant task frames. Meant for debugging."""
	def __init__(self):
		self.__doc__ = 'Prints list of tasks.\usage: ompd taskframes'
		super(ompd_taskframes, self).__init__('ompd taskframes',
					gdb.COMMAND_STACK)
	
	def invoke(self, arg, from_tty):
		frame = gdb.newest_frame()
		while(frame):
			print frame.read_register('sp')
			frame = frame.older()
		global addr_space
		curr_thread_handle = curr_thread().thread_handle
		curr_task_handle = ompdModule.call_ompd_get_curr_task_handle(curr_thread_handle)
		if(not curr_task_handle):
			return None
		
		try:
			while(1):
				frames_with_flags = ompdModule.call_ompd_get_task_frame(curr_task_handle)
				frames = (frames_with_flags[0], frames_with_flags[3])
				if(not isinstance(frames,tuple)):
					break
				(ompd_enter_frame, ompd_exit_frame) = frames
				print(hex(ompd_enter_frame), hex(ompd_exit_frame))
				curr_task_handle = ompdModule.call_ompd_get_scheduling_task_handle(curr_task_handle)
				if(not curr_task_handle):
					break
		except:
			traceback.print_exc()

def print_and_exec(string):
	"""Helper function for ompd_step. Executes the given command in GDB and prints it."""
	print(string)
	gdb.execute(string)

class TempFrameFunctionBp(gdb.Breakpoint):
	"""Helper class for ompd_step. Defines stop function for breakpoint on frame function."""
	def stop(self):
		global in_task_function
		in_task_function = True
		self.enabled = False

class ompd_step(gdb.Command):
	"""Executes 'step' and skips frames irrelevant to the application / the ones without debug information."""
	def __init__(self):
		self.__doc__ = 'Executes "step" and skips runtime frames as much as possible.'
		super(ompd_step, self).__init__('ompd step', gdb.COMMAND_STACK)
	
	class TaskBeginBp(gdb.Breakpoint):
		"""Helper class. Defines stop function for breakpoint ompd_bp_task_begin."""
		def stop(self):
			code_line = curr_thread().get_current_task().get_task_function()
			frame_fct_bp = TempFrameFunctionBp(('*%i' % code_line), temporary=True, internal=True)
			frame_fct_bp.thread = self.thread
			
			return False
	
	def invoke(self, arg, from_tty):
		global in_task_function
		
		tbp = self.TaskBeginBp('ompd_bp_task_begin', temporary=True, internal=True)
		tbp.thread = int(gdb.selected_thread().num)
		print_and_exec('step')
		
		while gdb.selected_frame().find_sal().symtab is None:
			if not in_task_function:
				print_and_exec('finish')
			else:
				print_and_exec('si')

def main():
	ompd()
	ompd_init()
	ompd_threads()
	ompd_test()
	ompd_taskframes()
	ompd_bt()
	ompd_step()

if __name__ == "__main__":
	try:
		main()
        except:
		traceback.print_exc()

# NOTE: test code using:
# OMP_NUM_THREADS=... gdb a.out -x ../../projects/gdb_plugin/gdb-ompd/__init__.py
# ompd init
# ompd threads
