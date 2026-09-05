/*"*********************************************************************************************************************
*                                                      LibTopoART                                                      *
*                                      created by Marko Tscherepanow, 12 June 2011                                     *
************************************************************************************************************************
*                                 $Id: LibTopoART.cs 1840 2026-08-15 10:43:24Z marko $                                 *
***********************************************************************************************************************/

using System;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Struct <c>CategoryInfo</c> summarises information about a node's category.</summary>
	public readonly struct CategoryInfo
	{
		/// <summary>Instance variable <c>spatial_weights</c> represents the spatial weights of the considered node.
		/// </summary>
		public readonly decimal[] spatial_weights;

		/// <summary>Instance variable <c>temporal_weights</c> represents the temporal weights of the considered node if
		/// it supports temporal learning.</summary>
		public readonly decimal[]? temporal_weights;

		/// <summary>Instance variable <c>clusterID</c> represents the cluster ID of the considered node.</summary>
		public readonly long clusterID;

		/// <summary>Instance variable <c>classID</c> represents the class ID of the considered node. If class IDs are
		/// not supported by the respective node, this value is set to <c>LibTopoART_info.UNDEFINED</c>.</summary>
		public readonly long classID;

		/// <summary>This constructor sets the instance variables <c>spatial_weights</c>, <c>temporal_weights</c>,
		/// <c>clusterID</c>, and <c>classID</c> of struct <c>CategoryInfo</c>.</summary>
		/// <param name="spatialWeights">The spatial weights to be set.</param>
		/// <param name="temporalWeights">The temporal weights to be set.</param>
		/// <param name="clusterID">The cluster ID to be set.</param>
		/// <param name="classID">The class ID to be set.</param>
		public CategoryInfo(decimal[] spatialWeights, decimal[]? temporalWeights, long clusterID, long classID)
		{
			spatial_weights = spatialWeights;
			temporal_weights = temporalWeights;
			this.clusterID = clusterID;
			this.classID = classID;
		}
	}

//**********************************************************************************************************************

	/// <summary>Struct <c>LibTopoART_control</c> provides fields to control the general behaviour of LibTopoART.
	/// </summary>
	public struct LibTopoART_control
	{
		/// <summary>Instance variable <c>verbosity</c> enables controlling the number of messages issued by LibTopoART.
		/// </summary>
		public static VerbosityLevel verbosity = VerbosityLevel.Verbose;
	}

//**********************************************************************************************************************

	/// <summary>Struct <c>LibTopoART_info</c> provides some metainformation regarding the respective implementation of
	/// LibTopoART.</summary>
	public readonly struct LibTopoART_info
	{
		/// <summary>Instance variable <c>version</c> represents the version of LibTopoART.</summary>
		public const decimal version = 1.01m;

		/// <summary>Instance variable <c>networks</c> provides a string array containing the networks implemented in
		/// the current version of LibTopoART and the corresponding class names.</summary>
		public static readonly string[] networks = [
			"Episodic TopoART (class Fast_Episodic_TopoART)",
			"Hypersphere TopoART (class Hypersphere_TopoART)",
			"Hypersphere TopoART-C (class Hypersphere_TopoART_C)",
			"TopoART (class TopoART, class Fast_TopoART)",
			"TopoART-AM (class Fast_TopoART_AM)",
			"TopoART-C (class TopoART_C, class Fast_TopoART_C)",
			"TopoART-R (class TopoART_R, class Fast_TopoART_R)"
		];

		/// <summary>Instance variable <c>UNDEFINED</c> gives the value used for indicating undefined and uninitialised
		/// variables.</summary>
		public const long UNDEFINED = -1;

		/// <summary>Instance variable <c>FINAL_MODULE</c> gives the value used for indicating that the TopoART module
		/// with the highest index is to be used.</summary>
		public const long FINAL_MODULE = UNDEFINED;

		/// <summary>Instance variable <c>maximum_threads</c> sets the maximum number of threads to be applied.
		/// </summary>
		internal static readonly long MaximumThreads = 2 * Environment.ProcessorCount;
	}

//**********************************************************************************************************************

	/// <summary>Class <c>F2_output</c> provides the output of a single TopoART module. It is a compressed version of
	/// the output vectors y and c.</summary>
	public class F2_output
	{
		/// <summary>Instance variable <c>bm_node_activation</c> represents the activation of the best-matching node
		/// (prediction variant).</summary>
		public decimal bm_node_activation = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>bm_node_ID</c> represents the ID of the best-matching node.</summary>
		public long bm_node_ID = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>bm_cluster_ID</c> represents the cluster ID of the best-matching node.
		/// </summary>
		public long bm_cluster_ID = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>bm_permanent_node_activation</c> represents the activation of the
		/// best-matching permanent node (prediction variant).</summary>
		public decimal bm_permanent_node_activation = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>bm_permanent_node_ID</c> represents the ID of the best-matching permanent
		/// node.</summary>
		public long bm_permanent_node_ID = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>bm_permanent_cluster_ID</c> represents the cluster ID of the best-matching
		/// permanent node.</summary>
		public long bm_permanent_cluster_ID = LibTopoART_info.UNDEFINED;
	}

//**********************************************************************************************************************

	/// <summary>Struct <c>TopoART_C_prediction</c> contains a prediction made by a TopoART-C network.</summary>
	public readonly struct TopoART_C_prediction
	{
		/// <summary>Instance variable <c>classID</c> gives the predicted class ID.</summary>
		public readonly long classID;

		/// <summary>Instance variable <c>confidence</c> provides a confidence for the predicted class ID.</summary>
		public readonly decimal confidence;

		/// <summary>This constructor sets the instance variables <c>classID</c> and <c>confidence</c> of struct
		/// <c>TopoART_C_prediction</c>.</summary>
		/// <param name="classID">The class ID to be set.</param>
		/// <param name="confidence">The value of the confidence to be set.</param>
		public TopoART_C_prediction(long classID, decimal confidence)
		{
			this.classID = classID;
			this.confidence = confidence;
		}
	}

//**********************************************************************************************************************

	/// <summary>Struct <c>TopoART_R_prediction</c> contains a prediction made by a TopoART-R network.</summary>
	public readonly struct TopoART_R_prediction<TElementType> where TElementType : struct, IConvertible
	{
		private static readonly TElementType _noPrediction =
			(typeof(TElementType) == typeof(decimal)) ? (TElementType)(object)(decimal)LibTopoART_info.UNDEFINED : default;

		/// <summary>Instance variable <c>NO_PREDICTION</c> provides a default prediction for variables that are
		/// presented to the network; i.e., these variables are known and no prediction is computed for them. ATTENTION:
		/// <c>NO_PREDICTION</c> may be ambiguous depending on <c>TElementType</c>.</summary>
		public readonly TElementType NO_PREDICTION;

		/// <summary>Instance variable <c>i_vec_prediction</c> represents predictions for unknown independent variables.
		/// </summary>
		public readonly TElementType[] i_vec_prediction;
		/// <summary>Instance variable <c>d_vec_prediction</c> provides the predictions for the dependent variables.
		/// </summary>
		public readonly TElementType[] d_vec_prediction;

		/// <summary>This constructor sets the instance variables <c>i_vec_prediction</c> and <c>d_vec_prediction</c> of
		/// struct <c>TopoART_R_prediction</c>.</summary>
		/// <param name="iVecPrediction">The prediction results for the independent variables to be set.</param>
		/// <param name="dVecPrediction">The prediction results for the dependent variables to be set.</param>
		public TopoART_R_prediction(TElementType[] iVecPrediction, TElementType[] dVecPrediction)
		{
			NO_PREDICTION = _noPrediction;

			i_vec_prediction = iVecPrediction;
			d_vec_prediction = dVecPrediction;
		}

		/// <summary>This constructor initialises the instance variables <c>i_vec_prediction</c> and
		/// <c>d_vec_prediction</c> of struct <c>TopoART_R_prediction</c> with arrays of the given lengths in which all
		/// elements are set to <c>NO_PREDICTION</c>.</summary>
		/// <param name="iLen">The length of the input vector (independent variables).</param>
		/// <param name="dLen">The length of the output vector (dependent variables).</param>
		public TopoART_R_prediction(long iLen, long dLen)
			: this(new TElementType[iLen], new TElementType[dLen])
		{
			for(long i = 0; i < iLen; ++i)
				i_vec_prediction[i] = NO_PREDICTION;
			for(long i = 0; i < dLen; ++i)
				d_vec_prediction[i] = NO_PREDICTION;
		}

		/// <summary>This method prints the predictions on the console.</summary>
		public void PrintPredictions() {
			Console.Write("i_vec_prediction:");
			foreach(TElementType prediction in i_vec_prediction)
				Console.Write($" {prediction}");
			Console.Write("\n");

			Console.Write("d_vec_prediction:");
			foreach(TElementType prediction in d_vec_prediction)
				Console.Write($" {prediction}");
			Console.Write("\n");
		}
	}

//**********************************************************************************************************************

	/// <summary>Enumeration specifying possible adaptation states.</summary>
	[Flags]
	public enum AdaptationState
	{
		/// <summary>No adaptation occurred.</summary>
		NO_ADAPTATION						=	0,
		/// <summary>Added one or more node candidates.</summary>
		ADDED_NODE_CANDIDATE				=	0x0001,
		/// <summary>The change of at least a single weight of one node candidate exceeds the given threshold.</summary>
		ADAPTED_NONPERMANENT_WEIGHT			=	0x0002,
		/// <summary>Added an edge from/to a node candidate.</summary>
		ADDED_EDGE_CANDIDATE				=	0x0004,
		/// <summary>Removed one or more node candidates.</summary>
		REMOVED_NODE_CANDIDATE				=	0x0008,
		/// <summary>Removed one or more node candidates.</summary>
		REMOVED_EDGE_CANDIDATE				=	0x0010,
		/// <summary>Mask for all non-permanent adaptations.</summary>
		ANY_NONPERMANENT_ADAPTATION_MASK	=	0x00ff,
		/// <summary>Added one or more permanent nodes.</summary>
		ADDED_PERMANENT_NODE				=	0x0100,
		/// <summary>The change of at least a single weight of one permanent node exceeds the given threshold.</summary>
		ADAPTED_PERMANENT_WEIGHT			=	0x0200,
		/// <summary>Added an edge between two permanent nodes.</summary>
		ADDED_PERMANENT_EDGE				=	0x0400,
		/// <summary>Mask for all permanent adaptations.</summary>
		ANY_PERMANENT_ADAPTATION_MASK		=	0xff00
	}

//**********************************************************************************************************************

	/// <summary>Enumeration specifying possible adaptation states.</summary>
	public enum VerbosityLevel : uint
	{
		/// <summary>Enables only the most important messages.</summary>
		Important,
		/// <summary>Enables the standard messages.</summary>
		Normal,
		/// <summary>Enables all messages.</summary>
		Verbose
	}

//**********************************************************************************************************************

	/// <summary>Exception signalling an invalid class ID.</summary>
	public class InvalidClassIDException : Exception
	{
		internal InvalidClassIDException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid file.</summary>
	public class InvalidFileException : Exception
	{
		internal InvalidFileException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid module index.</summary>
	public class InvalidModuleIndexException : Exception
	{
		internal InvalidModuleIndexException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid number.</summary>
	public class InvalidNumberException : Exception
	{
		internal InvalidNumberException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid size.</summary>
	public class InvalidSizeException : Exception
	{
		internal InvalidSizeException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid state of the neural network.</summary>
	public class InvalidStateException : Exception
	{
		internal InvalidStateException(string message) : base(message) {}
	}

	/// <summary>Exception signalling an invalid type.</summary>
	public class InvalidTypeException : Exception
	{
		internal InvalidTypeException(string message) : base(message) {}
	}

//**********************************************************************************************************************

}