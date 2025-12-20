using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Collections.Generic;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal class TopoART_module : 
		ActivationThreadProvider<decimal, decimal, long, bool>, 
		ITopoART_module<decimal, decimal, long, bool>,
		IModuleAdaptationStateCheck<decimal>
	{
		private long _x_F1_len;
		private decimal[]? _x_F1;
		private decimal _rho;
		private long _nextID;
		internal long _nodeNum;
		private TA_F2_node? _nodes;
		private bool _validClusterIDs;
		internal long _clusterNum;
		private long _learningCycles;

		private CreateF2Node<TA_F2_node, decimal, long>? _F2_node_create_func;

		private Snapshot<decimal>? _stateSnapshot;

		private bool _disposed;

//----------------------------------------------------------------------------------------------------------------------

		public List<CategoryInfo>? Categories { get => Common.GetCategoryInfos(_nodeNum, _nodes); }
		internal CreateF2Node<TA_F2_node, decimal, long>? CreateF2NodeFunction { set => _F2_node_create_func = value; }
		public long LearningCycles { get => _learningCycles; }

//----------------------------------------------------------------------------------------------------------------------

		// Do not use!
		private protected TopoART_module() {}

		public TopoART_module(long inputLen, decimal rho, CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func)
		{
			CreateF2NodeFunction = F2_node_create_func;

			_x_F1_len = inputLen;
			_x_F1 = null;
			_rho = rho;
			_nextID = 0;
			_nodeNum = 0;
			_nodes = null;
			_validClusterIDs = false;
			_clusterNum = 0;
			_learningCycles = 0;

			InitThreads();

			Common.Message($"Create module (x_F1_len = {_x_F1_len}; rho = {_rho:0.##########})");
		}

		public TopoART_module(BinaryReader reader, FileFormatVersions fileFormatVersions,
			CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func, LoadF2Node<TA_F2_node> F2_node_load_func)
		{
			CreateF2NodeFunction = F2_node_create_func;
			LoadTopoARTModuleParams(reader, fileFormatVersions);
			LoadNodes(reader, fileFormatVersions, F2_node_load_func);
		}

//----------------------------------------------------------------------------------------------------------------------

		~TopoART_module()
		{
			Dispose(false);
		}

		protected override void Dispose(bool disposing)
		{
			if(!_disposed) {
				if(disposing) 
					StopThreads();
				_disposed = true;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private void LoadTopoARTModuleParams(BinaryReader reader, FileFormatVersions fileFormatVersions)
		{
			_x_F1_len = reader.ReadInt64();
			var available = reader.ReadBoolean();
			if(!available) {
				_x_F1	=	null;
			} else {
				_x_F1	=	new decimal[_x_F1_len];
				for(long i = 0; i < _x_F1_len; ++i)
					_x_F1[i] = reader.ReadDecimal();
			}

			_rho = reader.ReadDecimal();
			_nextID = reader.ReadInt64();
			_validClusterIDs = reader.ReadBoolean();
			_clusterNum = reader.ReadInt64();
			_learningCycles = reader.ReadInt64();
			_nodeNum = reader.ReadInt64();

			InitThreads();
		}

		private void LoadNodes(BinaryReader reader, FileFormatVersions fileFormatVersions, 
			LoadF2Node<TA_F2_node> F2_node_load_func)
		{
			long i;
			TA_F2_node? previousNode;

			for(i = 0, previousNode = null; i < _nodeNum; ++i) {
				var currentNode = F2_node_load_func(reader, (fileFormatVersions, true));

				if(i == 0) {
					_nodes = currentNode;
				} else if (previousNode != null) {
					previousNode._next = currentNode;
				}
				previousNode = currentNode;

				// insert into thread node list determined by i
				var threadID =	i % LibTopoART_info.MaximumThreads;
				InsertIntoThreadArray(currentNode, threadID);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public void RemoveNodeCandidates (long phi)
		{
			TA_F2_node? currentNode, previousNode;
			var removalIDs = new long[_nodeNum];
			long removalNum = 0;
			
			for(currentNode = _nodes, previousNode = null; currentNode != null; currentNode = currentNode._next) {
				if(currentNode.IsNodeCandidate(phi)) {
					removalIDs[removalNum] = currentNode.NodeID;
					++removalNum;
					--_nodeNum;

					// remove from node list
					if(previousNode == null)
						_nodes = currentNode._next;
					else
						previousNode._next = currentNode._next;

					RemoveFromThreadArray(currentNode);
				}
				else
					previousNode = currentNode;
			}
			
			for(currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				for(long i = 0; i < removalNum; ++i)
					currentNode.RemoveEdgeTo(removalIDs[i]);
			}
		}

		public void ComputeClusterIDs(long phi)
		{
			_validClusterIDs = true;
			_clusterNum = Common.ComputeClusterIDs(_nodes, phi);
		}

		private decimal GetMinActivation(decimal mu, decimal sigma)
		{
			return mu + 1.281m * sigma;			// original: mu + 1.28 * sigma
		}

		// This function needs to be copied in Release configuration.
		public bool ComputeAlternativeChoiceFunctionsWithMaskAndNu(decimal[] x_F1, bool[]? mask, long nu,
			out Stack<TA_F2_node> enclosingNodes, out List<TA_F2_node> neighbouringNodes)
		{
			Debug.Assert(nu != 0);

			enclosingNodes = new Stack<TA_F2_node>(); 
			neighbouringNodes = new List<TA_F2_node>((int)nu);

			// if no nodes exist
			if(_nodes == null) 
				return false;

			RunActivationThreads(0.0m, mask, x_F1, ActivationType.Prediction);

			TA_F2_node? currentNode;
			long i;
			var mu = 0.0m;
			var sigma = 0.0m;
			
			for(currentNode = _nodes, i = 0; currentNode != null; currentNode = currentNode._next) {
				decimal activation = currentNode.Activation;
				mu		+=	activation;
				sigma	+=	(activation * activation);
				++i;

				if(activation == 1.0m)
					enclosingNodes.Push(currentNode);
			}

			if(i != 0) {
				mu		/=	i;
				sigma	/=	i;
				sigma	-=	(mu * mu);
				sigma	=	(decimal)Math.Sqrt((double)sigma);
			}

			// if no nodes fit
			if(mu == -1.0m)
				return false;

#if !DEBUG
			if(enclosingNodes.Count == 0) {
#endif
				var minActivation = GetMinActivation(mu, sigma);
				int j;
				
				for(currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
					if(currentNode.Activation != 1.0m) {
						if(neighbouringNodes.Count < nu)
							neighbouringNodes.Add(currentNode);
#if NET472_OR_GREATER
						else if(currentNode.Activation > neighbouringNodes[neighbouringNodes.Count - 1].Activation)
							neighbouringNodes[neighbouringNodes.Count - 1] = currentNode;
#else
						else if(currentNode.Activation > neighbouringNodes[^1].Activation)
							neighbouringNodes[^1] = currentNode;
#endif
						else
							continue;

						for(j = neighbouringNodes.Count - 1; j > 0; --j) {
							if(neighbouringNodes[j].Activation > neighbouringNodes[j - 1].Activation)
								(neighbouringNodes[j - 1], neighbouringNodes[j]) = (neighbouringNodes[j], neighbouringNodes[j - 1]);
						}
					}
				}

				for(j = 1; j < neighbouringNodes.Count; ++j) {				// keep at least one node
					if(neighbouringNodes[j].Activation <= minActivation)	// original <
						break;
				}
				if((neighbouringNodes.Count > 1) && (j < neighbouringNodes.Count))
					neighbouringNodes.RemoveRange(j, neighbouringNodes.Count - j);
#if !DEBUG
			}
#endif

			return true;
		}

		public IF2_node<decimal, decimal, long, bool>? GetBMNode()
		{
			TA_F2_node? bmNode = null;
			var maxActivation = -1.0m;

			for(var currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				if(currentNode.Activation > maxActivation) {
					bmNode = currentNode;
					maxActivation = currentNode.Activation;
				}
			}

			return bmNode;
		}

		public F2_output GetBMOutputWithMask(decimal[] input, bool[]? mask, long phi)
		{
			if(!_validClusterIDs)
				ComputeClusterIDs(phi);

			RunActivationThreads(0.0m, mask, input, ActivationType.Prediction);

			var result = new F2_output();		// init activations with -1.0m
			for(var currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				var activation = currentNode.Activation;
				if(activation > result.bm_node_activation) {
					result.bm_node_activation = activation;
					result.bm_node_ID = currentNode.NodeID;
					result.bm_cluster_ID = currentNode.ClusterID;
				}
				if((activation > result.bm_permanent_node_activation) && (!currentNode.IsNodeCandidate(phi))) {
					result.bm_permanent_node_activation = activation;
					result.bm_permanent_node_ID = currentNode.NodeID;
					result.bm_permanent_cluster_ID = currentNode.ClusterID;
				}
			}

			return result;
		}

		public LearningResult LearnWithMask(decimal[] x_F1_vec, bool[]? mask, MatchFunction<TA_F2_node, decimal>? matchFunction, 
			decimal alpha, decimal beta_sbm, long phi, bool skipEdgeLearning)
		{
			_x_F1 = x_F1_vec;
			TA_F2_node? bmNode;

			++_learningCycles;

			_validClusterIDs = false;

			if(_nodes == null) {
				bmNode = AddNode(_x_F1);
				goto finish;
			}

			RunActivationThreads(alpha, mask, _x_F1);

			bmNode = null;
			var bmActivation = -1.0m;
			TA_F2_node?  sbmNode = null;
			var sbmActivation = -1.0m;
			for(var currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				var currentActivation = currentNode.GetCombinedChoiceAndMatchValue(matchFunction, _rho);

				// check if bm needs to be changed
				if(currentActivation > bmActivation) {
					// bm is suitable as sbm
					sbmNode = bmNode;
					sbmActivation = bmActivation;
					// set new bm
					bmNode = currentNode;
					bmActivation = currentActivation;
				}
				// check if sbm needs to be changed
				else if(currentActivation > sbmActivation) {
					sbmNode = currentNode;
					sbmActivation = currentActivation;
				}
			}

			if(bmNode == null) {
				bmNode = AddNode(_x_F1);
				goto finish;
			} 

			bmNode.AdaptWeights(_x_F1, 1.0m);
			
			if(sbmNode != null) {
				sbmNode.AdaptWeights(_x_F1, beta_sbm);
				if(!skipEdgeLearning) {
					bmNode.AddEdgeTo(sbmNode.NodeID);
					sbmNode.AddEdgeTo(bmNode.NodeID);
				}
			}

finish:
			if(bmNode == null)
				return LearningResult.DoNotPropagate;

			if(bmNode.IsNodeCandidate(phi)) 
				return LearningResult.DoNotPropagate;

			return LearningResult.PropagateFurther;
		}

		private protected virtual TA_F2_node AddNode(decimal[] weights)
		{
			Debug.Assert(weights.LongLength == _x_F1_len, "Incompatible weight vector size.");
			Debug.Assert(_F2_node_create_func != null);

			// get thread_id before node counter is incremented
			var threadID = _nodeNum % LibTopoART_info.MaximumThreads;

			var newNode = _F2_node_create_func!(_nextID, _x_F1_len, weights, null);
			++_nextID;
			++_nodeNum;
			newNode._next = _nodes;
			_nodes = newNode;

			InsertIntoThreadArray(newNode, threadID);

			return newNode;
		}

//----------------------------------------------------------------------------------------------------------------------

		public void SaveText(TextWriter writer)
		{
			writer.WriteLine("x^F1 length: " + _x_F1_len);
			writer.Write("x^F1:");
			if(_x_F1 == null) {
				writer.WriteLine(" null");
			} else {
				for(long i = 0; i < _x_F1_len; ++i)
					writer.Write(" " + _x_F1[i].ToString(CultureInfo.InvariantCulture));

				writer.Write("\n");
			}

			writer.WriteLine("rho: " + _rho.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("next ID: " + _nextID);
			writer.WriteLine("valid cluster IDs: " + (_validClusterIDs ? "true" : "false"));
			writer.WriteLine("cluster number: " + _clusterNum);
			writer.WriteLine("learning cycles: " + LearningCycles);
			writer.WriteLine("node number: " + _nodeNum);
			for(var currentNode = _nodes; currentNode != null;) {
				currentNode.SaveText(writer);
				currentNode = currentNode._next;
			}
		}

		public void Save(BinaryWriter writer)
		{
			writer.Write(_x_F1_len);

			if(_x_F1 == null) {
				writer.Write(false);
			} else {
				writer.Write(true);
				for(long i = 0; i < _x_F1_len; ++i)
					writer.Write(_x_F1[i]);
			}

			writer.Write(_rho);
			writer.Write(_nextID);
			writer.Write(_validClusterIDs);
			writer.Write(_clusterNum);
			writer.Write(LearningCycles);
			writer.Write(_nodeNum);
			for(var currentNode = _nodes; currentNode != null;) {
				currentNode.Save(writer);
				currentNode = currentNode._next;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public void ResetAdaptationState(long phi)
		{
			_stateSnapshot = new Snapshot<decimal>(_nodes, phi);
		}

		public AdaptationState GetAdaptationState(long phi, CompareWeights<decimal> weightCmpFunction)
		{
			var currentStateSnapshot = new Snapshot<decimal>(_nodes, phi);
			_stateSnapshot ??= new Snapshot<decimal>(null, phi);
			return _stateSnapshot.CompareTo(currentStateSnapshot, weightCmpFunction);
		}
	}

//**********************************************************************************************************************

}