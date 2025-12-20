using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal sealed class Fast_Episodic_TopoART_module : Fast_TopoART_module 
	{
		private long[]? _t_F1;

		private F3_node? _F3_nodes;
		private FETA_F2_node? _active_F2_node;
		private long _interRecallCount;

		// required for fast access to nodes by their ID during recall
		private Dictionary<long, FTA_F2_node>? _F2_nodes_dictionary;

//----------------------------------------------------------------------------------------------------------------------

		public Fast_Episodic_TopoART_module(long inputLen, int rho, 
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
			: base(inputLen, rho, F2_node_create_func)
		{
			_t_F1  = null;
			ResetRecallMembers();
		}

		public Fast_Episodic_TopoART_module(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func, LoadF2Node<FTA_F2_node> F2_node_load_func)
			: base(reader, fileFormatInfo, F2_node_create_func, F2_node_load_func)
		{
			ResetRecallMembers();
		}

//----------------------------------------------------------------------------------------------------------------------

		private void ResetRecallMembers()
		{
			_F3_nodes = null;
			_active_F2_node = null;
			_F2_nodes_dictionary = null;
			_interRecallCount = 0;
		}

//----------------------------------------------------------------------------------------------------------------------

		public LearningResult LearnWithMask(Vector<int>[] x_F1_vec, long[] t_F1_vec, 
			MatchFunction<FTA_F2_node, long>? matchFunction, int alpha, int beta_sbm, long phi,
			bool skipEdgeLearning)
		{
			_x_F1 = x_F1_vec;
			_t_F1 = t_F1_vec;

			++_learningCycles;

			_validClusterIDs = false;

			FETA_F2_node? bmNode;
			if(_nodes == null) {
				bmNode = AddNode(_x_F1, _t_F1);
				goto finish;
			}

			RunActivationThreads(alpha, null, _x_F1, _t_F1);

			// find best-matching node for increased vigilance
			bmNode = null;
			var bmActivation = (int)-Common.ScalingFactor;

			for(var currentNode = (FETA_F2_node)_nodes; currentNode != null; currentNode = (FETA_F2_node?)currentNode._next) {
				var currentActivation = currentNode.GetCombinedChoiceAndMatchValue(matchFunction, (_rho + (int)Common.ScalingFactor) >> 1);
				if(currentActivation > bmActivation) {
					bmNode = currentNode;
					bmActivation = currentActivation;
				}
			}

			if(bmNode == null) {
				bmNode = AddNode(_x_F1, _t_F1);
				bmNode.ComputeChoiceAndMatchFunction(_x_F1, _t_F1, null, alpha);
			}

			// temporarily reset the bm activation to find the sbm node
			var tmpBmMatchValue = bmNode.MatchValue;
			bmNode.MatchValue = (int)-Common.ScalingFactor;

			// find second best-matching node for normal vigilance
			FETA_F2_node? sbmNode = null;
			var sbmActivation = (int)-Common.ScalingFactor;
			for(var currentNode = (FETA_F2_node)_nodes; currentNode != null; currentNode = (FETA_F2_node?)currentNode._next) {
				var currentActivation = currentNode.GetCombinedChoiceAndMatchValue(matchFunction, _rho);
				if(currentActivation > sbmActivation) {
					sbmNode = currentNode;
					sbmActivation = currentActivation;
				}
			}

			// restore bm activation
			bmNode.MatchValue = tmpBmMatchValue;

			// a best-matching node is always found
			bmNode.AdaptWeights(_x_F1, _t_F1, (int)Common.ScalingFactor);

			if(sbmNode != null) {
				sbmNode.AdaptWeights(_x_F1, _t_F1, beta_sbm);
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

		private FETA_F2_node AddNode(Vector<int>[] spatialWeights, long[] temporalWeights)
		{
			Debug.Assert(_F2_node_create_func != null);
			Debug.Assert(spatialWeights.LongLength == Common.SimdLengthEncoded<int>(_x_F1_len), "Incompatible spatial weight vector size.");
			Debug.Assert(temporalWeights.LongLength == 2, "Incompatible temporal weight vector size.");

			// get thread_id before node counter is incremented
			var threadID = _nodeNum % LibTopoART_info.MaximumThreads;

			var newNode = (FETA_F2_node)_F2_node_create_func!(_nextID, _x_F1_len, spatialWeights, temporalWeights);
			++_nextID;
			++_nodeNum;
			newNode._next = _nodes;
			_nodes = newNode;

			InsertIntoThreadArray(newNode, threadID);

			return newNode;
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void LoadPrecedingBinaryInformation(BinaryReader reader, 
			  (FileFormatVersions fileFormatVersions, bool compatibilityMode) fileFormatInfo)
		{
			base.LoadPrecedingBinaryInformation(reader, fileFormatInfo);

			if(fileFormatInfo.fileFormatVersions.FileFormatVersion == 0.01m)
				reader.ReadInt64();

			var available = reader.ReadBoolean();
			if(!available)
				_t_F1 = null;
			else {
				_t_F1 = new long[2];

				if(fileFormatInfo.compatibilityMode) {
					_t_F1[0] = (long)(reader.ReadDecimal() * Common.ScalingFactor);
					_t_F1[1] = (long)(reader.ReadDecimal() * Common.ScalingFactor);
				} else {
					_t_F1[0] = reader.ReadInt64();
					_t_F1[1] = reader.ReadInt64();
				}
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SavePrecedingTextInformation(TextWriter writer)
		{
			base.SavePrecedingTextInformation(writer);

			writer.Write("t^F1:");
			if(_t_F1 == null)
				writer.WriteLine(" null");
			else {
				writer.Write(" " + (_t_F1[0] / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
				writer.Write(" " + (_t_F1[1] / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
			}
			writer.Write("\n");
		}

		private protected override void SavePrecedingBinaryInformation(BinaryWriter writer, bool compatibilityMode)
		{
			base.SavePrecedingBinaryInformation(writer, compatibilityMode);
#if DEBUG
			if(Common.Episodic_TopoART_file_format_version == 0.01m)
				writer.Write(Common.t_max_test_v001);
#endif
			if(_t_F1 == null)
				writer.Write(false);
			else {
				writer.Write(true);
				if(compatibilityMode) {
					writer.Write(_t_F1[0] / (decimal)Common.ScalingFactor);
					writer.Write(_t_F1[1] / (decimal)Common.ScalingFactor);
				} else {
					writer.Write(_t_F1[0]);
					writer.Write(_t_F1[1]);
				}
			}
		}

//----------------------------------------------------------------------------------------------------------------------

#if DEBUG
		/// <summary>This method starts the recall process.</summary>
		/// <param name="complementCodedStimulus">The complement-coded stimulus (input) which is used to trigger recall.
		/// </param>
		/// <param name="phi">The parameter phi required for the removal of nodes and edges as well as for the
		/// propagation of input to subsequent TopoART modules.</param>
		/// <returns>The number of F3 nodes created.</returns>
#endif
		internal long BeginRecall(Vector<int>[] complementCodedStimulus, long phi)
		{
			if(!_validClusterIDs)
				ComputeClusterIDs(phi);

			ComputeAlternativeChoiceFunctionsWithMask(complementCodedStimulus, null);

			var F3_node_num = _clusterNum;

			if(F3_node_num > 0) {
				ActivateF3Nodes(F3_node_num, ref _F2_nodes_dictionary, ref _F3_nodes);

				_active_F2_node = (FETA_F2_node?)_F3_nodes?.BestMatchingF2Node;
				_interRecallCount = 0;
			} else
				ResetRecallMembers();

			return F3_node_num;
		} 

#if DEBUG
		/// <summary>This method performs a single inter-episode recall step and sets the starting point for
		/// intra-episode recall.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool InterEpisodeRecallStep(out byte[]? recallResult, out decimal F3_activation)
		{
			var result = InterEpisodeRecallStepCommon(out F3_activation);
			recallResult = _F3_nodes?.BestMatchingF2Node?.ByteAccessCentreOfGravity;
			Debug.Assert((result && recallResult != null)  || (!result && recallResult == null && F3_activation == LibTopoART_info.UNDEFINED));
			return result;
		}

#if DEBUG
		/// <summary>This method performs a single inter-episode recall step and sets the starting point for
		/// intra-episode recall.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool InterEpisodeRecallStep(out decimal[]? recallResult, out decimal F3_activation)
		{
			var result = InterEpisodeRecallStepCommon(out F3_activation);
			recallResult = _F3_nodes?.BestMatchingF2Node?.DecimalAccessCentreOfGravity;
			Debug.Assert((result && recallResult != null)  || (!result && recallResult == null && F3_activation == LibTopoART_info.UNDEFINED));
			return result;
		}

		private bool InterEpisodeRecallStepCommon(out decimal F3_activation)
		{
			bool result;

			if((_F3_nodes != null) && (_interRecallCount != 0)) 
				_F3_nodes = _F3_nodes._next;

			if(_F3_nodes != null) {
				F3_activation = _F3_nodes.Activation / (decimal)Common.ScalingFactor;
				_active_F2_node = (FETA_F2_node?)_F3_nodes.BestMatchingF2Node;
				result = true;
			} else {
				F3_activation = LibTopoART_info.UNDEFINED;
				_active_F2_node = null;
				result = false;
			}

			++_interRecallCount;

			return result;
		}

#if DEBUG
		/// <summary>This method performs a single intra-episode recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool IntraEpisodeRecallStep(out byte[]? recallResult)
		{
			bool result;
			var bmNode = IntraEpisodeRecallStepCommon();

			if(bmNode != null) {
				recallResult = bmNode.ByteAccessCentreOfGravity;
				_active_F2_node = (FETA_F2_node)bmNode;
				result = true;
			} else {
				recallResult = null;
				result = false;
			}

			Debug.Assert((result && recallResult != null) || (!result && recallResult == null));

			return result;
		}

#if DEBUG
		/// <summary>This method performs a single intra-episode recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool IntraEpisodeRecallStep(out decimal[]? recallResult)
		{
			bool result;
			var bmNode = IntraEpisodeRecallStepCommon();

			if(bmNode != null) {
				recallResult = bmNode.DecimalAccessCentreOfGravity;
				_active_F2_node = (FETA_F2_node)bmNode;
				result = true;
			} else {
				recallResult = null;
				result = false;
			}

			Debug.Assert((result && recallResult != null) || (!result && recallResult == null));

			return result;
		}

		private FTA_F2_node? IntraEpisodeRecallStepCommon()
		{
			FTA_F2_node? bmNode = null;
			var bmTemporalDist = 0.0m;

			if(_F2_nodes_dictionary != null) {
				if((_active_F2_node != null) && (_interRecallCount != 0)) {
					_active_F2_node.GetConnectedNodeIDs(out var size, out var connectedNodeIDs);
					for(long i = 0; i < size; ++i) {
						FTA_F2_node? currentNode;
						try {
							currentNode = _F2_nodes_dictionary[connectedNodeIDs![i]];
						} catch {
							Debug.WriteLine("Incorrect F2 node searched during intra-episode recall");
							continue;
						}

						decimal currentTemporalDist = _active_F2_node.ComputeForwardTemporalDistance((FETA_F2_node)currentNode);

						if(currentTemporalDist > bmTemporalDist) {
							bmNode = currentNode;
							bmTemporalDist = currentTemporalDist;
						} 
					}
				}
			}

			return bmNode;
		}

#if DEBUG
		/// <summary>This method stops the recall process and frees temporary resources.</summary>
#endif
		internal void EndRecall()
		{
			ResetRecallMembers();
		}
	}

//**********************************************************************************************************************

}