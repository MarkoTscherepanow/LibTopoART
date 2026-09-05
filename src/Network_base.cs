using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Network_base</c> provides the functionality required by all neural network implementations of
	/// LibTopoART.</summary>
	public abstract class Network_base
	{

//----------------------------------------------------------------------------------------------------------------------

		private protected class LearningTaskQueue
		{
			public Queue<Task> _tasks = new Queue<Task>();
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Instance variable <c>x_F0_len</c> represents the input length.</summary>
		private protected long _x_F0_len;

		private long _tau;
		private protected long[]? _phis;

		private protected bool _skipEdgeLearning = false;

		/// <summary>Instance variable <c>FINAL_MODULE</c> gives the value used for indicating that the TopoART module
		/// with the highest index is to be used.</summary>
		public const long FINAL_MODULE = LibTopoART_info.FINAL_MODULE;

		private protected LearningTaskQueue _learningTasks = new LearningTaskQueue();

#if NET9_0_OR_GREATER
		private protected readonly Lock _learningLock = new Lock();
#else
		private protected readonly object _learningLock = new object();
#endif

//----------------------------------------------------------------------------------------------------------------------

		/// <value>Property <c>InputLen</c> returns the length of the input vector.</value>
		public long InputLen
		{
			get => _x_F0_len;
		}

		/// <value>Property <c>LearningSteps</c> represents the total number of performed learning steps.</value>
		public long LearningSteps
		{
			get;
			private protected set;
		}

		/// <value>Property <c>ModuleNum</c> represents the number of TopoART modules used. (The original TopoART uses
		/// two modules.)</value>
		public long ModuleNum
		{
			get;
			private protected set;
		}

		/// <value>Property <c>Phi</c> represents the parameter phi required for the removal of nodes and edges as well
		/// as for the propagation of input to subsequent TopoART modules.</value>
		public long Phi
		{
			get {
				long commonPhi;
				if(_phis == null)
					commonPhi = LibTopoART_info.UNDEFINED;
				else {
					commonPhi = _phis[0];
					for(long i = 1; i < _phis.LongLength; ++i) {
						if(_phis[i] != commonPhi)
							commonPhi = LibTopoART_info.UNDEFINED;
					}
				}
				return commonPhi;
			}
			set {
				if(LearningSteps == 0) {
					var tmpPhi = value;
					if(tmpPhi < 1) {
						tmpPhi = 1;
						Common.Warning("Too small value for phi, changed to " + tmpPhi);
					}
					Common.Message("phi set to " + tmpPhi);
					if(tmpPhi > (_tau / 10))
						Common.Warning("phi might be too large in comparison to tau");
					_phis ??= new long[ModuleNum];
					for(long i = 0; i < _phis.LongLength; ++i)
						_phis[i] = tmpPhi;
				}
				else
					Common.Warning("Unable to set phi after training started, keep old value " + Phi);
			}
		}

		/// <value>Property <c>Phis</c> constitutes an extension of property <c>Phi</c> that enables individual values
		/// of phi for each module. By this, the removal of nodes and edges as well as the propagation of input to
		/// subsequent TopoART modules can be controlled in a task-dependent manner.</value>
		/// <exception cref="InvalidSizeException">Thrown when the array length does not fit the module number.
		/// </exception>
		public long[] Phis
		{
			get {
				if(_phis != null)
					return (long[])_phis.Clone();
				else {
					var tmp = new long[ModuleNum];
					for(long i = 0; i < ModuleNum; ++i)
						tmp[i] = LibTopoART_info.UNDEFINED;
					return tmp;
				}
			}
			set {
				lock(_learningLock) {
					CompleteLearningQueue();

					if(value.LongLength != ModuleNum)
						throw new InvalidSizeException(Common.InvalidSizeException_InvalidPhiArrayLength);

					if(LearningSteps == 0) {
						var tmpPhis = (long[])value.Clone();

						for(long i = 0; i < ModuleNum; ++i) {
							var subscript = (char)('a' + i);
							if(tmpPhis[i] < 1) {
								tmpPhis[i] = 1;
								Common.Warning("Too small value for phi_" + subscript + ", changed to " + tmpPhis[i]);
							}
							Common.Message("phi_" + subscript + " set to " + tmpPhis[i]);
							if(tmpPhis[i] > (_tau / 10))
								Common.Warning("phi_" + subscript + " might be too large in comparison to tau", VerbosityLevel.Verbose);
						}

						_phis = tmpPhis;
					} else {
						if(_phis != null) {
							string warning = "Unable to set phis after training started, keep old values " + _phis[0];
							for(long i = 1; i < _phis.LongLength; ++i)
								warning += ", " + _phis[i];
							Common.Warning(warning);
						} else
							Common.Warning("Unable to set phis after training started");
					}
				}
			}
		}

		/// <value>Property <c>Tau</c> represents the parameter tau required for the removal of nodes and edges.</value>
		public long Tau
		{
			get => _tau;
			set {
				if(LearningSteps == 0) {
					_tau = value;
					if(_tau < 1) {
						_tau = 1;
						Common.Warning("Too small value for tau, changed to " + Tau);
					}
					Common.Message("tau set to " + Tau);
				} else
					Common.Warning("Unable to set tau after training started, keep old value " + Tau);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected static long CheckLength(long length)
		{
			return (length < 1) ? 1 : length;
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected void CompleteLearningQueue()
		{
			while(_learningTasks._tasks.Count > 0)
				_learningTasks._tasks.Dequeue().Wait();
		}

		private protected void CompleteLearningQueueNoThrow()
		{
			try {
				CompleteLearningQueue();
			} catch(AggregateException e) {
				Common.Warning("Suppressed exception of a faulted learning task: " + (e.InnerException ?? e).Message);
			}
		}

		private protected void Learn(long moduleNum, ModuleFunction moduleFunction)
		{
			lock(_learningLock) {
				CompleteLearningQueue();

				_learningTasks._tasks.Enqueue(Task.Factory.StartNew(() => {
						for(long i = 0; i < moduleNum; ++i)
							moduleFunction(i);
				}));
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method resets the adaptation state to <c>AdaptationState.NO_ADAPTATION</c>.</summary>
		/// <exception cref="InvalidNumberException">Thrown when the number of edges of an F2 node is greater than
		/// <c>int.MaxValue</c>.</exception>
		private protected void ResetAdaptationState<TModuleType, TFloatType>(TModuleType[]? modules)
			where TModuleType : IModuleAdaptationStateCheck<TFloatType>
		{
			Debug.Assert(modules != null);

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i)
					modules![i].ResetAdaptationState(_phis![i]);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected abstract void SaveTextHeader(TextWriter writer);
		private protected abstract void SavePrecedingTextInformation(TextWriter writer);
	}

//**********************************************************************************************************************

}