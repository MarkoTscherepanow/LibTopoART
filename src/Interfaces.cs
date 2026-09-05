using System;
using System.IO;
using System.Collections.Generic;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	internal interface IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>
	{
		long ClassID { get; }
		long ClusterID { get; set; }
		IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? Next { get; }
		long NodeID { get; }

		void ComputeAlternativeChoiceFunction(TSpatialWeightType[] x_F1, TMaskType[]? mask);
		void ComputeChoiceAndMatchFunction(TSpatialWeightType[] x_F1, TTemporalWeightType[]? t_F1,
										   TMaskType[]? mask, TFloatType alpha);
		void GetConnectedNodeIDs(out long size, out long[]? connectedNodeIDs);
		decimal[] GetCopyOfSpatialWeights();
		decimal[]? GetCopyOfTemporalWeights();
		bool IsNodeCandidate(long phi);
	}

//**********************************************************************************************************************

	internal interface IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType> :
		IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>
	{
		IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? ThreadNext { get; set; }
		IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? ThreadPrev { get; set; }
		long ThreadID { get; set; }
	}

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Interface summarising the methods required to assess the adaptation state of a node.</summary>
	public interface IF2_node_state<TSpatialWeightType>
#else
	internal interface IF2_node_state<TSpatialWeightType>
#endif
	{
		/// <value>Property <c>StateNext</c> provides a reference to the state of the next F2 node.</value>
		IF2_node_state<TSpatialWeightType>? StateNext { get; }

		/// <value>Property <c>StateNext</c> provides the ID of an F2 node.</value>
		long NodeID { get; }

		/// <value>Property <c>Edges</c> provides the edges of an F2 node as a list.</value>
		List<(long, long)> Edges { get; }

		/// <value>Property <c>Weights</c> provides the weights of an F2 node as an array.</value>
		TSpatialWeightType[] Weights { get; }

		/// <summary>This method checks whether a node is a node candidate with respect to a given value of
		/// <paramref name="phi"/>.</summary>
		/// <param name="phi">The value of phi to be used.</param>
		bool IsNodeCandidate(long phi);
	}

//**********************************************************************************************************************

	internal interface IModuleAdaptationStateCheck<TFloatType>
	{
		void ResetAdaptationState(long phi);
		AdaptationState GetAdaptationState(long phi, CompareWeights<TFloatType> weightCmpFunction);
	}

//**********************************************************************************************************************

	internal interface ITopoART_module<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>
	{
		List<CategoryInfo>? Categories { get; }
		long LearningCycles { get; }

		void RemoveNodeCandidates (long phi);
		void ComputeClusterIDs(long phi);
		IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? GetBMNode();
		F2_output GetBMOutputWithMask(TSpatialWeightType[] input, TMaskType[] mask, long phi);
		void SaveText(TextWriter writer);
	}

//**********************************************************************************************************************

	/// <summary>Interface providing access to the learnt categories, e.g., for drawing.</summary>
	public interface ICategoryAccess
	{
		/// <summary>This method collects information on the categories of a specified module.</summary>
		/// <param name="moduleIndex">The index of the module information on the categories of which is to be returned.
		/// </param>
		/// <returns>A list containing information about the respective categories.</returns>
		List<CategoryInfo>? GetCategories(long moduleIndex = LibTopoART_info.FINAL_MODULE);
	}

//**********************************************************************************************************************

	/// <summary>Interface enabling checks whether certain adaptations of a network occurred.</summary>
	public interface IAdaptationStateCheck
	{
		/// <summary>This method resets the adaptation state to <c>AdaptationState.NO_ADAPTATION</c>.</summary>
		void ResetAdaptationState();

		/// <summary>This method returns the current adaptation state.</summary>
		/// <param name="epsilon">The threshold for weight adaptations to be considered.</param>
		/// <returns>An enumeration describing the adaptation state.</returns>
		AdaptationState GetAdaptationState(decimal epsilon = 0.001m);
	}

//**********************************************************************************************************************

	/// <summary>Interface summarising the type-independent functionality to stop the recall process.</summary>
	public interface IEndRecall
	{
		/// <summary>This method stops the recall process and frees temporary resources.</summary>
		void EndRecall();
	}

//**********************************************************************************************************************

	/// <summary>Interface providing access to the basic associative recall functionality using stimulus elements and
	/// recall result elements of type <c>TAccessType</c>.</summary>
	public interface IAccessAssociativeRecall<TAccessType> : IEndRecall
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method starts the recall process for the first key vector.</summary>
		/// <param name="key2">The stimulus (second key vector) which is used to trigger recall.</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall.
		/// (<c>LibTopoART_info.FINAL_MODULE</c> denotes the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		long BeginRecallKey1(TAccessType[] key2, long moduleIndex = LibTopoART_info.FINAL_MODULE);

		/// <summary>This method starts the recall process for the second key vector.</summary>
		/// <param name="key1">The stimulus (first key vector) which is used to trigger recall.</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall.
		/// (<c>LibTopoART_info.FINAL_MODULE</c> denotes the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		long BeginRecallKey2(TAccessType[] key1, long moduleIndex = LibTopoART_info.FINAL_MODULE);

		/// <summary>This method performs a single associative recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
		bool RecallStep(out TAccessType[]? recallResult, out decimal F3_activation);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the associative recall functionality using stimulus elements and recall result
	/// elements of type <c>decimal</c>.</summary>
	public interface IAssociativeRecall : IAccessAssociativeRecall<decimal> {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the associative recall functionality using stimulus elements and recall result
	/// elements of type <c>byte</c> or of type <c>decimal</c>.</summary>
	public interface IFastAssociativeRecall : IAssociativeRecall, IAccessAssociativeRecall<byte> {}

//**********************************************************************************************************************

	/// <summary>Interface providing access to the basic episodic recall functionality using stimulus elements and
	/// recall result elements of type <c>_AccessType</c>.</summary>
	public interface IAccessEpisodicRecall<TAccessType> : IEndRecall
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method starts the recall process.</summary>
		/// <param name="stimulus">The stimulus (input) which is used to trigger recall.</param>
		/// <returns>The number of F3 nodes created.</returns>
		long BeginRecall(TAccessType[] stimulus);

		/// <summary>This method performs a single inter-episode recall step and sets the starting point for
		/// intra-episode recall.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
		bool InterEpisodeRecallStep(out TAccessType[]? recallResult, out decimal F3_activation);

		/// <summary>This method performs a single intra-episode recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed or not.</returns>
		bool IntraEpisodeRecallStep(out TAccessType[]? recallResult);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the episodic recall functionality using stimulus elements and recall result
	/// elements of type <c>decimal</c>.</summary>
	public interface IEpisodicRecall : IAccessEpisodicRecall<decimal> {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the episodic recall functionality using stimulus elements and recall result
	/// elements of type <c>byte</c> or of type <c>decimal</c>.</summary>
	public interface IFastEpisodicRecall : IEpisodicRecall, IAccessEpisodicRecall<byte> {}

//**********************************************************************************************************************

	/// <summary>Interface summarising the basic TopoART functionality excluding learning and prediction.</summary>
	public interface ITopoART_base
	{
		/// <value>Property <c>InputLen</c> returns the length of the input vector.</value>
		public long InputLen { get; }

		/// <value>Property <c>NodeNum</c> represents the number of TopoART nodes used by each module.</value>
		long[] NodeNum { get; }

		/// <value>Property <c>ClusterNum</c> represents the number of TopoART clusters found by each module.</value>
		long[] ClusterNum { get; }

		/// <value>Property <c>ModuleNum</c> represents the number of TopoART modules used. (The original TopoART uses
		/// two modules.)</value>
		long ModuleNum { get; }

		/// <value>Property <c>LearningSteps</c> represents the total number of performed learning steps.</value>
		long LearningSteps { get; }

		/// <value>Property <c>Beta_sbm</c> represents the learning rate of the second best-matching nodes.</value>
		decimal Beta_sbm { get; set; }

		/// <value>Property <c>Rho_a</c> represents the vigilance parameter of the first TopoART module (TA a).</value>
		decimal Rho_a { get; }

		/// <value>Property <c>Tau</c> represents the parameter tau required for the removal of nodes and edges.</value>
		long Tau { get; set; }

		/// <value>Property <c>Phi</c> represents the parameter phi required for the removal of nodes and edges as well
		/// as for the propagation of input to subsequent TopoART modules.</value>
		long Phi { get; set; }

		/// <value>Property <c>Phis</c> constitutes an extension of property <c>Phi</c> that enables individual values
		/// of phi for each module. By this, the removal of nodes and edges as well as the propagation of input to
		/// subsequent TopoART modules can be controlled in a task-dependent manner.</value>
		long[] Phis { get; set; }

		/// <value>Property <c>Alpha</c> represents the choice parameter alpha.</value>
		decimal Alpha { get; set; }

		/// <summary>This method computes the cluster IDs for all neurons.</summary>
		void ComputeClusterIDs();

		/// <summary>This method saves the entire network as a text file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		void SaveText(string path);

		/// <summary>This method saves the entire network as a binary file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		/// <param name="compression">Compression level of the save file (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		void Save(string path, CompressionLevel compression = CompressionLevel.Fastest);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface extending the basic TopoART functionality by stream-based saving.</summary>
	public interface ITopoART_base_stream : ITopoART_base
	{
		/// <summary>This method saves the entire network as text to a writer. The writer is flushed but left open.
		/// </summary>
		/// <param name="writer">A <c>TextWriter</c> the network is saved to.</param>
		void SaveText(TextWriter writer);

		/// <summary>This method saves the entire network to a stream using the binary file format. The stream is left
		/// open.</summary>
		/// <param name="stream">A writable <c>Stream</c> the network is saved to.</param>
		/// <param name="compression">Compression level of the saved data (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		void Save(Stream stream, CompressionLevel compression = CompressionLevel.Fastest);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface providing access to the basic TopoART functionality using input elements of type
	/// <c>_AccessType</c>.</summary>
	public interface IAccess_TopoART<TAccessType> : ITopoART_base
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t).</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		F2_output[] GetBMOutput(TAccessType[] input);

		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t).</param>
		/// <param name="mask">A mask vector excluding individual dimensions of x(t) from the computation. (Setting an
		/// element of the mask vector to <c>true</c>, excludes the corresponding elements of x(t).)</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		F2_output[] GetBMOutput(TAccessType[] input, bool[] mask);

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		void Learn(TAccessType[] input);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART functionality including learning and prediction using input elements
	/// of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface ITopoART : IAccess_TopoART<decimal>, IAdaptationStateCheck {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART functionality including learning and prediction using input elements
	/// of type <c>byte</c> or of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface IFast_TopoART : ITopoART, IAccess_TopoART<byte> {}

//**********************************************************************************************************************

	/// <summary>Interface summarising the Episodic TopoART functionality including learning, prediction, episodic
	/// recall using input elements, stimulus elements, and recall result elements of type <c>decimal</c> as well as
	/// adaptation state control.</summary>
	public interface IEpisodic_TopoART :  ITopoART, IEpisodicRecall
	{
		/// <value>Property <c>T_max</c> represents the maximum considered time frame.</value>
		long T_max { get; }
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the Episodic TopoART functionality including learning, prediction, episodic
	/// recall using input elements, stimulus elements, and recall result elements of type <c>byte</c> or of type
	/// <c>decimal</c> as well as adaptation state control.</summary>
	public interface IFast_Episodic_TopoART : IEpisodic_TopoART, IFast_TopoART, IFastEpisodicRecall {}

//**********************************************************************************************************************

	/// <summary>Interface summarising the Hypersphere TopoART functionality including learning and prediction using
	/// input elements of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface IHypersphere_TopoART : ITopoART
	{
		/// <value>Property <c>R</c> represents the radial extend parameter R.</value>
		decimal R { get; }
	}

//**********************************************************************************************************************

	/// <summary>Interface summarising the basic TopoART-AM functionality excluding learning and prediction.</summary>
	public interface ITopoART_AM_base : ITopoART_base
	{
		/// <summary>Property <c>Key1Len</c> returns the length of the first key vector.</summary>
		long Key1Len { get; }

		/// <summary>Property <c>Key2Len</c> returns the length of the second key vector.</summary>
		long Key2Len  {get; }
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface providing access to the basic TopoART-AM functionality using input elements of type
	/// <c>_AccessType</c>.</summary>
	public interface IAccess_TopoART_AM<TAccessType> : ITopoART_AM_base
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method finds the closest category for a given pair of keys.</summary>
		/// <param name="key1">The first key vector.</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>.</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART-AM module.</returns>
		F2_output[] GetBMOutput(TAccessType[] key1, TAccessType[] key2);

		/// <summary>This method performs a single training step.</summary>
		/// <param name="key1">The first key vector to be learnt.</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>.</param>
		void Learn(TAccessType[] key1, TAccessType[] key2);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART-AM functionality including learning, prediction, associative recall
	/// using input elements, stimulus elements, and recall result elements of type <c>decimal</c> as well as adaptation
	/// state control.</summary>
	public interface ITopoART_AM : ITopoART, IAccess_TopoART_AM<decimal>, IAssociativeRecall {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the Episodic TopoART functionality including learning, prediction, episodic
	/// recall using input elements, stimulus elements, and recall result elements of type <c>byte</c> or of type
	/// <c>decimal</c> as well as adaptation state control.</summary>
	public interface IFast_TopoART_AM : ITopoART_AM, IFast_TopoART, IAccess_TopoART_AM<byte>, IFastAssociativeRecall {}

//**********************************************************************************************************************

	/// <summary>Interface summarising the basic TopoART-C functionality excluding learning and prediction.</summary>
	public interface ITopoART_C_base : ITopoART_base
	{
		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the set of
		/// enclosing categories E and the neighbourhood set N during prediction. If the parameter <c>nu</c> is not
		/// explicitly provided for prediction, this property will be applied. (This parameter does not modify the
		/// network. It may be arbitrarily changed for each prediction step.)</summary>
		public long Nu { get; set; }

		/// <summary>Property <c>SkipEdgeLearning</c> enables/disables the TopoART edge learning mechanism. If the
		/// topology of the input data is not required, disabling edge learning may decrease the processing time needed
		/// for training.</summary>
		public bool SkipEdgeLearning { get; set; }
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface providing access to the basic TopoART-C functionality using input elements of type
	/// <c>_AccessType</c>.</summary>
	public interface IAccess_TopoART_C<TAccessType> : ITopoART_C_base
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		/// <param name="classID">The class ID corresponding to <paramref name="input"/>.</param>
		void Learn(TAccessType[] input, long classID);

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <returns>The predicted class ID.</returns>
		long Predict(TAccessType[] input);

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted class ID.</returns>
		long Predict(TAccessType[] input, long nu);

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		TopoART_C_prediction Predict(TAccessType[] input, bool[] mask);

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		TopoART_C_prediction Predict(TAccessType[] input, bool[] mask, long nu);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART-C functionality including learning and prediction using input
	/// elements of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface ITopoART_C : ITopoART, IAccess_TopoART_C<decimal> {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART-C functionality including learning and prediction using input
	/// elements of type <c>byte</c> or of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface IFast_TopoART_C : ITopoART_C, IFast_TopoART, IAccess_TopoART_C<byte> {}

//**********************************************************************************************************************

	/// <summary>Interface summarising the basic TopoART-R functionality excluding learning and prediction.</summary>
	public interface ITopoART_R_base : ITopoART_base
	{
		/// <summary>Property <c>D_len</c> returns the length of the output vector (dependent variables).</summary>
		long D_len { get; }

		/// <summary>Property <c>I_len</c> returns the length of the input vector (independent variables).</summary>
		long I_len { get; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the
		/// neighbourhood set N during prediction. If the parameter <c>nu</c> is not explicitly provided for prediction,
		/// this property will be applied. (This parameter does not modify the network. It may be arbitrarily changed
		/// for each prediction step.)</summary>
		public long Nu { get; set; }

		/// <summary>Property <c>SkipEdgeLearning</c> enables/disables the TopoART edge learning mechanism. If the
		/// topology of the input data is not required, disabling edge learning may decrease the processing time needed
		/// for training.</summary>
		public bool SkipEdgeLearning { get; set; }
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface providing access to the basic TopoART-R functionality using input elements of type
	/// <c>_AccessType</c>.</summary>
	public interface IAccess_TopoART_R<TAccessType> : ITopoART_R_base
		where TAccessType : struct, IConvertible
	{
		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector (independent variables) to be learnt.</param>
		/// <param name="output">The output vector (dependent variables) corresponding to <paramref name="input"/>.
		/// </param>
		void Learn(TAccessType[] input, TAccessType[] output);

		/// <summary>This method predicts the dependent variables using the default value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		TAccessType[] Predict(TAccessType[] input);

		/// <summary>This method predicts the dependent variables using a custom value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		TAccessType[] Predict(TAccessType[] input, long nu);

		/// <summary>This method predicts the dependent variables for a given set of independent variables using the
		/// default value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns>An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		TopoART_R_prediction<TAccessType> Predict(TAccessType[] input, bool[] mask);

		/// <summary>This method predicts the dependent variables for a given set of independent variables using a
		/// custom value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		TopoART_R_prediction<TAccessType> Predict(TAccessType[] input, bool[] mask, long nu);
	}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART-R functionality including learning and prediction using input
	/// elements and output elements of type <c>decimal</c> as well as adaptation state control.</summary>
	public interface ITopoART_R : ITopoART, IAccess_TopoART_R<decimal> {}

//----------------------------------------------------------------------------------------------------------------------

	/// <summary>Interface summarising the TopoART-R functionality including learning and prediction using input
	/// elements and output elements of type <c>byte</c> or of type <c>decimal</c> as well as adaptation state control.
	/// </summary>
	public interface IFast_TopoART_R : ITopoART_R, IFast_TopoART, IAccess_TopoART_R<byte> {}

//**********************************************************************************************************************

}