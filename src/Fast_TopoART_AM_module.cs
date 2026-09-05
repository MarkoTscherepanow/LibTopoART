using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Numerics;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal sealed class Fast_TopoART_AM_module : Fast_TopoART_module
	{
		private F3_node? _F3_nodes;
		private FTA_F2_node? _active_F2_node;
		private long _recallCount;

		// required for fast access to nodes by their ID during recall
		private Dictionary<long, FTA_F2_node>? _F2_nodes_dictionary;

//----------------------------------------------------------------------------------------------------------------------

		public Fast_TopoART_AM_module(long inputLen, int rho, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
			: base(inputLen, rho, F2_node_create_func)
		{
			ResetRecallMembers();
		}

		public Fast_TopoART_AM_module(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo,
									  CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func,
									  LoadF2Node<FTA_F2_node> F2_node_load_func)
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
			_recallCount = 0;
		}

//----------------------------------------------------------------------------------------------------------------------

#if DEBUG
		/// <summary>This method starts the recall process.</summary>
		/// <param name="complementCodedStimulus">The complement-coded stimulus (input) which is used to trigger recall.
		/// </param>
		/// <param name="mask">A mask vector excluding individual dimensions of x(t) from the computation. (Setting an
		/// element of the mask vector to <c>true</c>, excludes the corresponding elements of x(t).)</param>
		/// <param name="phi">The parameter phi required for the removal of nodes and edges as well as for the
		/// propagation of input to subsequent TopoART modules.</param>
		/// <returns>The number of F3 nodes created.</returns>
#endif
		internal long BeginRecall(Vector<int>[] complementCodedStimulus, Vector<int>[] mask, long phi)
		{
			if(!_validClusterIDs)
				ComputeClusterIDs(phi);

			ComputeAlternativeChoiceFunctionsWithMask(complementCodedStimulus, mask);

			var F3_node_num = _clusterNum;

			if(F3_node_num > 0) {
				ActivateF3Nodes(F3_node_num, ref _F2_nodes_dictionary, ref _F3_nodes);

				_active_F2_node = _F3_nodes?.BestMatchingF2Node;
				_recallCount = 0;
			} else
				ResetRecallMembers();

			return F3_node_num;
		}

#if DEBUG
		/// <summary>This method performs a single associative recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool RecallStep(out byte[]? recallResult, out decimal F3_activation)
		{
			var result = RecallStepCommon(out F3_activation);
			recallResult = _F3_nodes?.BestMatchingF2Node?.ByteAccessCentreOfGravity;
			Debug.Assert((result && recallResult != null)  || (!result && recallResult == null && F3_activation == LibTopoART_info.UNDEFINED));
			return result;
		}

#if DEBUG
		/// <summary>This method performs a single associative recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
#endif
		internal bool RecallStep(out decimal[]? recallResult, out decimal F3_activation)
		{
			var result = RecallStepCommon(out F3_activation);
			recallResult = _F3_nodes?.BestMatchingF2Node?.DecimalAccessCentreOfGravity;
			Debug.Assert((result && recallResult != null)  || (!result && recallResult == null && F3_activation == LibTopoART_info.UNDEFINED));
			return result;
		}

		private bool RecallStepCommon(out decimal F3_activation)
		{
			bool result;

			if((_F3_nodes != null) && (_recallCount != 0))
				_F3_nodes = _F3_nodes._next;

			if(_F3_nodes != null) {
				F3_activation = _F3_nodes.Activation / (decimal)Common.ScalingFactor;
				_active_F2_node = _F3_nodes.BestMatchingF2Node;
				result = true;
			} else {
				F3_activation = LibTopoART_info.UNDEFINED;
				_active_F2_node = null;
				result = false;
			}

			++_recallCount;

			return result;
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