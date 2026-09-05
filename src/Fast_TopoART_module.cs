using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using System.Numerics;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal class Fast_TopoART_module :
		ActivationThreadProvider<int, Vector<int>, long, Vector<int>>,
		ITopoART_module<int, Vector<int>, long, Vector<int>>,
		IModuleAdaptationStateCheck<Vector<int>>
	{
		private const long _serialWorkLimit = 32768;

		// per-node work offset (in SIMD-vector equivalents) accounting for element-count-independent costs
		private const long _serialPerNodeWorkOffset = 8;

		private protected long _x_F1_len;
		private protected Vector<int>[]? _x_F1;
		private protected int _rho;
		private protected long _nextID;
		internal long _nodeNum;
		private protected FTA_F2_node? _nodes;
		private protected bool _validClusterIDs;
		internal long _clusterNum;
		private protected long _learningCycles;

		private protected CreateF2Node<FTA_F2_node, Vector<int>, long>? _F2_node_create_func;

		private Snapshot<Vector<int>>? _stateSnapshot;

		private bool _disposed;

//----------------------------------------------------------------------------------------------------------------------

		public List<CategoryInfo>? Categories { get => Common.GetCategoryInfos(_nodeNum, _nodes); }

		internal CreateF2Node<FTA_F2_node, Vector<int>, long>? CreateF2NodeFunction
		{
			set => _F2_node_create_func = value;
		}

		public long LearningCycles { get => _learningCycles; }

//----------------------------------------------------------------------------------------------------------------------

		// Do not use!
		//private protected Fast_TopoART_module() {}

		public Fast_TopoART_module(long inputLen, int rho, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
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

			InitThreads(_serialWorkLimit, _serialPerNodeWorkOffset);

			Common.Message(string.Format("Create module (x_F1_len = {0}; rho = {1:0.##########})",
				_x_F1_len, _rho / (decimal)Common.ScalingFactor));
		}

		public Fast_TopoART_module(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func,
			LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			CreateF2NodeFunction = F2_node_create_func;
			LoadTopoARTModuleParams(reader, fileFormatInfo);
			LoadNodes(reader, fileFormatInfo, F2_node_load_func);
		}

//----------------------------------------------------------------------------------------------------------------------

		~Fast_TopoART_module()
		{
			Dispose(false);
		}

		protected override void Dispose(bool disposing)
		{
			if(!_disposed) {
				if(disposing)
					StopThreads();
				_disposed = true;

				base.Dispose(disposing);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private void LoadTopoARTModuleParams(BinaryReader reader, (FileFormatVersions, bool compatibilityMode) fileFormatInfo)
		{
			int[]? tmp_x_F1;

			// load additional information for derived networks
			LoadPrecedingBinaryInformation(reader, fileFormatInfo);

			_x_F1_len = reader.ReadInt64();
			var available = reader.ReadBoolean();
			if(!available) {
				tmp_x_F1 = null;
			} else {
				tmp_x_F1 = new int[_x_F1_len];
				if(fileFormatInfo.compatibilityMode) {
					for(long i = 0; i < _x_F1_len; ++i)
						tmp_x_F1[i] = (int)(reader.ReadDecimal() * Common.ScalingFactor);
					_rho = (int)(reader.ReadDecimal() * Common.ScalingFactor);
				} else {
					for(long i = 0; i < _x_F1_len; ++i)
						tmp_x_F1[i] = reader.ReadInt32();
					_rho = reader.ReadInt32();
				}
			}

			_x_F1 = Common.CreateEncodedVectorArray(tmp_x_F1);

			_nextID = reader.ReadInt64();
			_validClusterIDs = reader.ReadBoolean();
			_clusterNum = reader.ReadInt64();
			_learningCycles = reader.ReadInt64();
			_nodeNum = reader.ReadInt64();

			InitThreads(_serialWorkLimit, _serialPerNodeWorkOffset);
		}

		private void LoadNodes(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo,
			LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			long i;
			long threadID;
			FTA_F2_node? previousNode;

			for(i = 0, previousNode = null; i < _nodeNum; ++i) {
				var currentNode = F2_node_load_func(reader, fileFormatInfo);

				if(i == 0)
					_nodes = currentNode;
				else if (previousNode != null)
					previousNode._next = currentNode;
				previousNode = currentNode;

				// insert into thread node list determined by i
				threadID = i % LibTopoART_info.MaximumThreads;
				InsertIntoThreadArray(currentNode, threadID);
			}
		}

		private protected virtual void LoadPrecedingBinaryInformation(BinaryReader reader,
		 	(FileFormatVersions fileFormatVersions, bool compatibilityMode) fileFormatInfo) {}

//----------------------------------------------------------------------------------------------------------------------

		public void RemoveNodeCandidates (long phi)
		{
			var removalIDs = new long[_nodeNum];
			long removalNum = 0;

			for(FTA_F2_node? currentNode = _nodes, previousNode = null; currentNode != null; currentNode = currentNode._next) {
				if(currentNode.IsNodeCandidate(phi)) {
					removalIDs[removalNum] = currentNode.NodeID;
					++removalNum;
					--_nodeNum;

					// remove from node list
					if(previousNode == null) {
						_nodes = currentNode._next;
					} else {
						previousNode._next = currentNode._next;
					}

					RemoveFromThreadArray(currentNode);
				}
				else
					previousNode = currentNode;
			}

			for(var currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				for(long i = 0; i < removalNum; ++i)
					currentNode.RemoveEdgeTo(removalIDs[i]);
			}
		}

		public void ComputeClusterIDs(long phi)
		{
			_validClusterIDs = true;
			_clusterNum = Common.ComputeClusterIDs(_nodes, phi);
		}

		private long GetMinActivation(long mu, decimal sigma)
		{
			return mu + (long)(1.281m * sigma * Common.ScalingFactor);			// original: mu + 1.28 * sigma
		}

		protected void ComputeAlternativeChoiceFunctionsWithMask(Vector<int>[] x_F1, Vector<int>[]? mask)
		{
			RunActivationThreads(0, mask, x_F1, ActivationType.Prediction);
		}

		// This function needs to be copied in Release configuration.
		public bool ComputeAlternativeChoiceFunctionsWithMaskAndNu(Vector<int>[] x_F1, Vector<int>[]? mask, long nu,
			out Stack<FTA_F2_node>enclosingNodes, out List<FTA_F2_node> neighbouringNodes)
		{
			Debug.Assert(nu != 0);

			enclosingNodes = new Stack<FTA_F2_node>();
			neighbouringNodes = new List<FTA_F2_node>((int)nu);

			// if no nodes exist
			if(_nodes == null)
				return false;

			RunActivationThreads(0, mask, x_F1, ActivationType.Prediction);

			long i;
			long mu = 0;
			var sigma = 0.0m;
			FTA_F2_node? currentNode;
			for(currentNode = _nodes, i = 0; currentNode != null; currentNode = currentNode._next) {
				long activation = currentNode.Activation;
				mu		+=	activation;
				sigma	+=	(activation * activation);
				++i;

				if(activation == Common.ScalingFactor)
					enclosingNodes.Push(currentNode);
			}

			if(i != 0) {
				mu /= i;
				sigma /= i;
				sigma -= (mu * mu);
				sigma = (decimal)Math.Sqrt((double)((sigma / Common.ScalingFactor) / Common.ScalingFactor));
			}

			// if no nodes fit
			if(mu == -Common.ScalingFactor)
				return false;

#if !DEBUG
			if(enclosingNodes.Count == 0) {
#endif
				decimal minActivation = GetMinActivation(mu, sigma);
				int j;

				for(currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
					if(currentNode.Activation != Common.ScalingFactor) {
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
								(neighbouringNodes[j], neighbouringNodes[j - 1]) = (neighbouringNodes[j - 1], neighbouringNodes[j]);
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

		public IF2_node<int, Vector<int>, long, Vector<int>>? GetBMNode()
		{
			FTA_F2_node? bmNode = null;
			var maxActivation = -Common.ScalingFactor;

			for(var currentNode = _nodes; currentNode != null; currentNode = currentNode._next) {
				if(currentNode.Activation > maxActivation) {
					bmNode = currentNode;
					maxActivation = currentNode.Activation;
				}
			}

			return bmNode;
		}

		public F2_output GetBMOutputWithMask(Vector<int>[] input, Vector<int>[]? mask, long phi)
		{
			var result = new F2_output();		// init activations with -1.0m

			if(!_validClusterIDs)
				ComputeClusterIDs(phi);

			RunActivationThreads(0, mask, input, ActivationType.Prediction);

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

			if(result.bm_node_ID != LibTopoART_info.UNDEFINED)
				result.bm_node_activation /= Common.ScalingFactor;
			if(result.bm_permanent_node_ID != LibTopoART_info.UNDEFINED)
				result.bm_permanent_node_activation /= Common.ScalingFactor;

			return result;
		}

		public LearningResult LearnWithMask(Vector<int>[] x_F1_vec, Vector<int>[]? mask, MatchFunction<FTA_F2_node, long>? matchFunction,
			int alpha, int beta_sbm, long phi, bool skipEdgeLearning)
		{
			_x_F1 = x_F1_vec;
			FTA_F2_node? bmNode;

			++_learningCycles;

			_validClusterIDs = false;

			if(_nodes == null) {
				bmNode = AddNode(_x_F1);
				goto finish;
			}

			RunActivationThreads(alpha, mask, _x_F1);

			bmNode = null;
			var bmActivation = -Common.ScalingFactor;
			FTA_F2_node? sbmNode = null;
			var sbmActivation = -Common.ScalingFactor;
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
					sbmNode		=	currentNode;
					sbmActivation	=	currentActivation;
				}
			}

			if(bmNode == null) {
				bmNode = AddNode(_x_F1);
				goto finish;
			}

			bmNode.AdaptWeights(_x_F1, (int)Common.ScalingFactor);

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

		private FTA_F2_node AddNode(Vector<int>[] weights)
		{
			Debug.Assert(weights.LongLength == Common.SimdLengthEncoded<int>(_x_F1_len), "Incompatible weight vector size.");
			Debug.Assert(_F2_node_create_func != null);

			// get thread_id before node counter is incremented
			var threadId = _nodeNum % LibTopoART_info.MaximumThreads;

			var newNode = _F2_node_create_func!(_nextID, _x_F1_len, weights, null);
			++_nextID;
			++_nodeNum;
			newNode._next =	_nodes;
			_nodes = newNode;

			InsertIntoThreadArray(newNode, threadId);

			return newNode;
		}

//----------------------------------------------------------------------------------------------------------------------

		protected void ActivateF3Nodes(long F3_node_num, ref Dictionary<long, FTA_F2_node>? F2_nodes_dictionary,
			ref F3_node? F3_nodes)
		{
			var F3_nodes_array = new F3_node[F3_node_num];
			F2_nodes_dictionary = new Dictionary<long, FTA_F2_node>();

			// create a valid F3 node for each cluster
			for(var F2_nodes = _nodes; F2_nodes != null; F2_nodes = F2_nodes._next) {
				if(F2_nodes.ClusterID > 0) {
					// fill lookup dictionary
					F2_nodes_dictionary.Add(F2_nodes.NodeID, F2_nodes);

					// maximise F3 activation
					if(F3_nodes_array[F2_nodes.ClusterID - 1] == null)
						F3_nodes_array[F2_nodes.ClusterID - 1] = new F3_node(F2_nodes);
					else if(F2_nodes.Activation > F3_nodes_array[F2_nodes.ClusterID - 1].Activation)
						F3_nodes_array[F2_nodes.ClusterID - 1] = new F3_node(F2_nodes);
				}
			}

			// sort F3 nodes
			long i;
			for(i = 0, F3_nodes = null; i < F3_node_num; ++i)
			{
				F3_node? currentNode;
				F3_node? previousNode;

				for(previousNode = null, currentNode = F3_nodes; currentNode != null;
					previousNode = currentNode, currentNode = currentNode._next) {
					if(F3_nodes_array[i].Activation > currentNode.Activation)
						break;
				}

				if(previousNode == null) {
					F3_nodes_array[i]._next = F3_nodes;
					F3_nodes = F3_nodes_array[i];
				} else {
					F3_nodes_array[i]._next = previousNode._next;
					previousNode._next = F3_nodes_array[i];
				}
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		public void SaveText(TextWriter writer)
		{
			SavePrecedingTextInformation(writer);

			writer.WriteLine("x^F1 length: " + _x_F1_len);
			writer.Write("x^F1:");
			if(_x_F1 == null) {
				writer.WriteLine(" null");
			} else {
				for(long i = 0; i < _x_F1_len; ++i) {
					var (i1, i2) = Common.SimdIndexes<int>(i, _x_F1_len >> 1);
					writer.Write(" " + (_x_F1[i1][i2] / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
				}
				writer.Write("\n");
			}

			writer.WriteLine("rho: " + (_rho / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
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

		private protected virtual void SavePrecedingTextInformation(TextWriter writer)
		{
		}

		public void Save(BinaryWriter writer, bool compatibilityMode)
		{
			SavePrecedingBinaryInformation(writer, compatibilityMode);

			writer.Write(_x_F1_len);
			writer.Write(_x_F1 != null);

			if(compatibilityMode) {
				if(_x_F1 != null) {
					for(long i = 0; i < _x_F1_len; ++i) {
						var (i1, i2) = Common.SimdIndexes<int>(i, _x_F1_len >> 1);
						writer.Write(_x_F1[i1][i2] / (decimal)Common.ScalingFactor);
					}
				}
				writer.Write(_rho / (decimal)Common.ScalingFactor);
			} else {
				if(_x_F1 != null) {
					for(long i = 0; i < _x_F1_len; ++i) {
						var (i1, i2) = Common.SimdIndexes<int>(i, _x_F1_len >> 1);
						writer.Write(_x_F1[i1][i2]);
					}
				}
				writer.Write(_rho);
			}

			writer.Write(_nextID);
			writer.Write(_validClusterIDs);
			writer.Write(_clusterNum);
			writer.Write(LearningCycles);
			writer.Write(_nodeNum);
			for(var currentNode = _nodes; currentNode != null;) {
				currentNode.Save(writer, compatibilityMode);
				currentNode = currentNode._next;
			}
		}

		private protected virtual void SavePrecedingBinaryInformation(BinaryWriter writer, bool compatibilityMode) {}

//----------------------------------------------------------------------------------------------------------------------

		public void ResetAdaptationState(long phi)
		{
			_stateSnapshot = new Snapshot<Vector<int>>(_nodes, phi);
		}

		public AdaptationState GetAdaptationState(long phi, CompareWeights<Vector<int>> weightCmpFunction)
		{
			var currentStateSnapshot = new Snapshot<Vector<int>>(_nodes, phi);
			_stateSnapshot ??= new Snapshot<Vector<int>>(null, phi);
			return _stateSnapshot.CompareTo(currentStateSnapshot, weightCmpFunction);
		}
	}

//**********************************************************************************************************************

}