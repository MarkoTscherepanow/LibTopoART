/*"*********************************************************************************************************************
*                                                   TopoART-C classes                                                  *
*                                       created by Marko Tscherepanow, 25 June 2016                                    *
************************************************************************************************************************
*                                 $Id: TopoART_C.cs 1838 2026-07-17 21:32:05Z marko $                                  *
***********************************************************************************************************************/

using System;
using System.IO;
using System.Collections.Generic;
using System.Numerics;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>TopoART_C</c> provides an implementation of the TopoART-C neural network as proposed in
	/// "Marko Tscherepanow and Sören Riechers (2012). An Incremental On-line Classifier for Imbalanced, Incomplete, and
	/// Noisy Data. In Proceedings of the European Conference on Artificial Intelligence (ECAI), Workshop on Active and
	/// Incremental Learning (AIL), pp. 18-23. Montpellier, France."
	/// <para>Class <c>TopoART_C</c> requires all input except the class IDs to lie in the interval [0, 1]. The class
	/// IDs are signed integer values.</para>
	/// </summary>
	public class TopoART_C : TopoART, ITopoART_C
	{
		/// <summary>Instance variable <c>UNDEFINED</c> gives the value used for indicating undefined and uninitialised
		/// variables.</summary>
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>UNDEFINED_CLASS_ID</c> gives the value used for indicating that an input
		/// sample was predicted to belong to the undefined class; i.e., no class ID was provided for such input samples
		/// during training.</summary>
		public const long UNDEFINED_CLASS_ID = -2;

		private readonly string networkName = "TopoART-C";
		private const NetworkType networkType = NetworkType.TopoARTC;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>TopoART_C</c>.</summary>
		public new decimal FileFormatVersion { get => Common.TopoART_C_file_format_version; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the set of
		/// enclosing categories E and the neighbourhood set N during prediction. If the parameter <c>nu</c> is not
		/// explicitly provided for prediction, this property will be applied. (This parameter does not modify the
		/// network. It may be arbitrarily changed for each prediction step.)</summary>
		public long Nu
		{
			get => field;
			set
			{
				field = value;
				if (field < 1)
				{
					field = 1;
					Common.Warning("Too small value for nu, changed to " + Nu);
				}
				Common.Message("nu set to " + Nu);
			}
		} = Common.TopoART_C_default_nu;

		/// <summary>Property <c>SkipEdgeLearning</c> enables/disables the TopoART edge learning mechanism. If the
		/// topology of the input data is not required, disabling edge learning may decrease the processing time needed
		/// for training.</summary>
		public bool SkipEdgeLearning
		{
			get => _skipEdgeLearning;
			set {
				Common.Message((value ? "Disable" : "Enable") + " edge learning");
				_skipEdgeLearning = value;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This constructor initialises a TopoART-C network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART-C modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART-C module (TopoART-C a).</param>
		public TopoART_C(long inputLen, long moduleNum, decimal rho_a)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);
			InitModules(inputLen << 1, CreateTopoARTModule, null);
		}

		/// <summary>This constructor loads a saved TopoART-C network.</summary>
		/// <param name="path">The path of a binary TopoART-C file.</param>
		/// <exception cref="InvalidFileException">Thrown when the given file cannot be loaded.</exception>
		public TopoART_C(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			LoadNetwork(file);
		}

		/// <summary>This constructor loads a saved TopoART-C network from a stream. The stream is left open.</summary>
		/// <param name="stream">A readable <c>Stream</c> containing a network in the binary TopoART-C file format.
		/// </param>
		/// <exception cref="InvalidFileException">Thrown when the given stream cannot be loaded.</exception>
		public TopoART_C(Stream stream)
		{
			LoadNetwork(stream);
		}

		private void LoadNetwork(Stream stream)
		{
			using var outerReader = LoadTopoARTParams(stream, null, out var fileFormatVersions);
			InitModules(outerReader, fileFormatVersions, LoadTopoARTModule, null, (BinaryReader reader,
				in (FileFormatVersions localFileFormatVersions, bool) localFileFormatInfo) =>
					new TAC_F2_node(reader, localFileFormatInfo.localFileFormatVersions));
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step and sets the class ID corresponding to
		/// <paramref name="input"/> to <c>UNDEFINED_CLASS_ID</c>.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			LearnInternal(input, UNDEFINED_CLASS_ID);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		/// <param name="classID">The class ID corresponding to <paramref name="input"/>. (must be equal to or larger
		/// than 0)</param>
		/// <exception cref="InvalidClassIDException">Thrown when <paramref name="classID"/> is less than 0.</exception>
		public void Learn(decimal[] input, long classID)
		{
			if(classID < 0)
				throw new InvalidClassIDException(Common.InvalidClassIDException_NegativeClassID);
			else
				LearnInternal(input, classID);
		}

		private void LearnInternal(decimal[] input, long classID)
		{
			CreateF2Node<TA_F2_node, decimal, long> createFunction =
				(nodeID, inputLen, spatialWeights, temporalWeights) =>
				{
					Debug.Assert(temporalWeights == null);
					return(new TAC_F2_node(nodeID, inputLen, spatialWeights, classID));
				};

			Debug.Assert(_modules != null);

			MatchFunction<TA_F2_node, decimal> matchFunction = (node, rho) =>
				node.MatchValue >= rho && ((TAC_F2_node)node).ClassID == classID;

			LearnWithMask(input, null, createFunction, matchFunction);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(decimal[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(decimal[] input, long nu)
		{
			return Predict(input, null, nu).classID;
		}

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(decimal[] input, bool[]? mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(decimal[] input, bool[]? mask, long nu)
		{
			lock(_learningLock) {
				CompleteLearningQueue();

				Debug.Assert(_modules != null);
				Debug.Assert(_x_F0 != null);

				var classID = UNDEFINED;
				var confidence = 1.0m;

				if(_modules![ModuleNum - 1]._nodeNum >= 1) {		// net not empty
					if(nu < 1) {
						nu = 1;
						Common.Warning("Invalid value for nu, changed to " + nu);
					}

					for(long i = 0; i < _x_F0_len; ++i)
						_x_F0![i] = input[i];

					var x_F1 = EncodeCurrentInput();

					if(_modules[ModuleNum - 1].ComputeAlternativeChoiceFunctionsWithMaskAndNu(
						x_F1, mask, nu, out Stack<TA_F2_node> enclosingNodes, out List<TA_F2_node> neighbouringNodes)) {
						if(enclosingNodes.Count > 0) {
							var minCategorySize = _x_F0_len + 0.0001m;
							long enclosingNodesCount = 0;
							do									// add at least one node
							{
								var currentNode = (TAC_F2_node)enclosingNodes.Pop();
								var tmpCategorySize = currentNode.Size;

								if(tmpCategorySize < minCategorySize) {
									minCategorySize = tmpCategorySize;
									classID = currentNode.ClassID;
								}

								++enclosingNodesCount;
							} while((enclosingNodes.Count > 0) && (enclosingNodesCount < nu));
						} else if(neighbouringNodes.Count > 0) {
							var invSum = 0.0m;
							var y_F2 = new decimal[neighbouringNodes.Count];
							for(long i = 0; i < neighbouringNodes.Count; ++i) {
								// division-by-zero is prevented by the previous branch
								y_F2[i] = 1.0m / (1.0m - neighbouringNodes[(int)i].Activation);
								invSum += y_F2[i];
							}

							// discrimination function: per-class sums in two small parallel buffers
							var classIDs = new long[neighbouringNodes.Count];
							var classSums = new decimal[neighbouringNodes.Count];
							var classNum = 0;

							for(long i = 0; i < neighbouringNodes.Count; ++i) {
								y_F2[i] /= invSum;

								var currentClassID = ((TAC_F2_node)neighbouringNodes[(int)i]).ClassID;
								var found = false;
								for(var j = 0; j < classNum; ++j) {
									if(classIDs[j] == currentClassID) {
										classSums[j] += y_F2[i];
										found = true;
										break;
									}
								}
								if(!found) {
									classIDs[classNum] = currentClassID;
									classSums[classNum] = y_F2[i];
									++classNum;
								}
							}

							confidence = neighbouringNodes[0].Activation;

							var maxDiscriminationValue = 0.0m;
							for(var j = 0; j < classNum; ++j) {
								if((classSums[j] > maxDiscriminationValue) ||
								   ((classSums[j] == maxDiscriminationValue) && (classIDs[j] < classID))) {
									maxDiscriminationValue = classSums[j];
									classID = classIDs[j];
								}
							}
						}
					}
				}

				return new TopoART_C_prediction(classID, confidence);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override (FileFormatVersions, SaveFlags) LoadBinaryHeader(BinaryReader reader)
		{
			var headerInfo = Common.LoadBinaryHeader(reader, networkType, networkName, new FileFormatVersions(FileFormatVersion,
				TopoARTFileFormatVersion), integerType, floatType);

			return (headerInfo.FileFormatVersions, headerInfo.Flags);
		}

		private protected override void LoadPrecedingBinaryInformation(BinaryReader reader, FileFormatVersions fileFormatVersions)
		{
			base.LoadPrecedingBinaryInformation(reader, fileFormatVersions);

			if(fileFormatVersions.FileFormatVersion >= 1.0m)
				Nu = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*       TopoART-C network       *");
			writer.WriteLine("*      LibTopoART (v" + LibTopoART_info.version.ToString(CultureInfo.InvariantCulture) + ")       *");
			writer.WriteLine("*********************************");
			writer.WriteLine("file format versions: " + FileFormatVersion.ToString(CultureInfo.InvariantCulture) + "; " + TopoARTFileFormatVersion.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("integer type: " + IntegerType);
			writer.WriteLine("float type: " + FloatType);
			writer.WriteLine("*********************************");
		}

		private protected override void SavePrecedingTextInformation(TextWriter writer)
		{
			base.SavePrecedingTextInformation(writer);
			writer.WriteLine("nu: " + Nu);
			writer.WriteLine("*********************************");
		}

		private protected override void SaveBinaryHeader(BinaryWriter writer, CompressionLevel compression)
		{
			Common.SaveBinaryHeader(writer, networkType, new FileFormatVersions(FileFormatVersion, TopoARTFileFormatVersion),
				(integerType, floatType), compression);
		}

		private protected override void SavePrecedingBinaryInformation(BinaryWriter writer)
		{
			base.SavePrecedingBinaryInformation(writer);
			writer.Write(Nu);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(networkName);
		}
	}

//**********************************************************************************************************************

	/// <summary>Class <c>Fast_TopoART_C</c> provides an implementation of the TopoART-C neural network as proposed in
	/// "Marko Tscherepanow and Sören Riechers (2012). An Incremental On-line Classifier for Imbalanced, Incomplete, and
	/// Noisy Data. In Proceedings of the European Conference on Artificial Intelligence (ECAI), Workshop on Active and
	/// Incremental Learning (AIL), pp. 18-23. Montpellier, France."
	/// <para>Internally, real-valued data are mapped to <c>int</c> variables. Therefore, computations are accelerated
	/// but less accurate. As a consequence, the results may differ slightly from class <c>TopoART_C</c>.</para>
	/// <para>Class <c>Fast_TopoART_C</c> requires all input except the class IDs to lie in the interval [0, 1]. The
	/// class IDs are signed integer values.</para>
	/// </summary>
	public class Fast_TopoART_C : Fast_TopoART, IFast_TopoART_C
	{
		/// <summary>Instance variable <c>UNDEFINED</c> gives the value used for indicating undefined and uninitialised
		/// variables.</summary>
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>UNDEFINED_CLASS_ID</c> gives the value used for indicating that an input
		/// sample was predicted to belong to the undefined class; i.e., no class ID was provided for such input samples
		/// during training.</summary>
		public const long UNDEFINED_CLASS_ID = -2;

		private long _nu = Common.TopoART_C_default_nu;
		private readonly string _networkName = "TopoART-C";
		private const NetworkType _networkType = NetworkType.TopoARTC;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Fast_TopoART_C</c>.</summary>
		public new decimal FileFormatVersion { get => Common.TopoART_C_file_format_version; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the set of
		/// enclosing categories E and the neighbourhood set N during prediction. If the parameter <c>nu</c> is not
		/// explicitly provided for prediction, this property will be applied. (This parameter does not modify the
		/// network. It may be arbitrarily changed for each prediction step.)</summary>
		public long Nu
		{
			get => _nu;
			set {
				_nu = value;
				if(_nu < 1) {
					_nu = 1;
					Common.Warning("Too small value for nu, changed to " + Nu);
				}
				Common.Message("nu set to " + Nu);
			}
		}

		/// <summary>Property <c>SkipEdgeLearning</c> enables/disables the TopoART edge learning mechanism. If the
		/// topology of the input data is not required, disabling edge learning may decrease the processing time needed
		/// for training.</summary>
		public bool SkipEdgeLearning
		{
			get => _skipEdgeLearning;
			set {
				_skipEdgeLearning = value;
				Common.Message((value ? "Disable" : "Enable") + " edge learning");
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This constructor initialises a TopoART-C network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART-C modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART-C module (TopoART-C a).</param>
		public Fast_TopoART_C(long inputLen, long moduleNum, decimal rho_a)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);
			InitModules(inputLen << 1, CreateTopoARTModule, null);
		}

		/// <summary>This constructor loads a saved TopoART-C network.</summary>
		/// <param name="path">The path of a binary TopoART-C file.</param>
		/// <exception cref="InvalidFileException">Thrown when the given file cannot be loaded.</exception>
		public Fast_TopoART_C(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			LoadNetwork(file);
		}

		/// <summary>This constructor loads a saved TopoART-C network from a stream. The stream is left open.</summary>
		/// <param name="stream">A readable <c>Stream</c> containing a network in the binary TopoART-C file format.
		/// </param>
		/// <exception cref="InvalidFileException">Thrown when the given stream cannot be loaded.</exception>
		public Fast_TopoART_C(Stream stream)
		{
			LoadNetwork(stream);
		}

		private void LoadNetwork(Stream stream)
		{
			using var outerReader = LoadTopoARTParams(stream, null, out var headerInfo);
			InitModules(outerReader, headerInfo, LoadTopoARTModule, null, (BinaryReader reader,
				in (FileFormatVersions, bool) localFileFormatInfo) => new FTAC_F2_node(reader, localFileFormatInfo));
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step and sets the class ID corresponding to
		/// <paramref name="input"/> to <c>UNDEFINED_CLASS_ID</c>.</summary>
		/// <param name="input">The input vector to be learnt. The input values are internally scaled from [0, 255] to
		/// [0, 1].</param>
		public override void Learn(byte[] input)
		{
			var (createFunction, matchFunction) = LearnCommon(UNDEFINED_CLASS_ID);
			LearnWithMask(input, null, createFunction, matchFunction);
		}

		/// <summary>This method performs a single training step and sets the class ID corresponding to
		/// <paramref name="input"/> to <c>UNDEFINED_CLASS_ID</c>.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			var (createFunction, matchFunction) = LearnCommon(UNDEFINED_CLASS_ID);
			LearnWithMask(input, null, createFunction, matchFunction);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt. The elements of the input vector are internally scaled
		/// from [0, 255] to [0, 1].</param>
		/// <param name="classID">The class ID corresponding to <paramref name="input"/>. (must be equal to or larger
		/// than 0)</param>
		/// <exception cref="InvalidClassIDException">Thrown when <paramref name="classID"/> is less than 0.</exception>
		public void Learn(byte[] input, long classID)
		{
			if(classID < 0)
				throw new InvalidClassIDException(Common.InvalidClassIDException_NegativeClassID);
			else {
				var (createFunction, matchFunction) = LearnCommon(classID);
				LearnWithMask(input, null, createFunction, matchFunction);
			}
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		/// <param name="classID">The class ID corresponding to <paramref name="input"/>. (must be equal to or larger
		/// than 0)</param>
		/// <exception cref="InvalidClassIDException">Thrown when <paramref name="classID"/> is less than 0.</exception>
		public void Learn(decimal[] input, long classID)
		{
			if(classID < 0)
				throw new InvalidClassIDException(Common.InvalidClassIDException_NegativeClassID);
			else {
				var (createFunction, matchFunction) = LearnCommon(classID);
				LearnWithMask(input, null, createFunction, matchFunction);
			}
		}

		private (CreateF2Node<FTA_F2_node, Vector<int>, long>, MatchFunction<FTA_F2_node, long>) LearnCommon(long classID)
		{
			CreateF2Node<FTA_F2_node, Vector<int>, long> createFunction =
				(nodeID, inputLen, spatialWeights, temporalWeights) =>
				{
					Debug.Assert(temporalWeights == null);
					return(new FTAC_F2_node(nodeID, inputLen, spatialWeights, classID));
				};

			Debug.Assert(_modules != null);

			return (createFunction, (node, rho) => node.MatchValue >= rho && ((FTAC_F2_node)node).ClassID == classID);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted. The elements of the input
		/// vector are internally scaled from [0, 255] to [0, 1].</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(byte[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(decimal[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted. The elements of the input
		/// vector are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(byte[] input, long nu)
		{
			return Predict(input, null, nu).classID;
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted class ID.</returns>
		public long Predict(decimal[] input, long nu)
		{
			return Predict(input, null, nu).classID;
		}

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted. The elements of the input
		/// vector are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(byte[] input, bool[]? mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(decimal[] input, bool[]? mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.
		/// The elements of the input vector are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(byte[] input, bool[]? mask, long nu)
		{
			return PredictCommon(() => {
						Debug.Assert(_x_F0_len == input.LongLength);
						EncodeCurrentInputSimd(input);
					}, mask, nu);
		}

		/// <summary>This method predicts the class ID using a custom value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the set of enclosing categories E and the neighbourhood set N.
		/// (This parameter does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(decimal[] input, bool[]? mask, long nu)
		{
			return PredictCommon(() => {
						Debug.Assert(_x_F0_len == input.LongLength);
						EncodeCurrentInputSimd(input);
					}, mask, nu);
		}

		private TopoART_C_prediction PredictCommon(Action encode, bool[]? mask, long nu)
		{
			lock(_learningLock) {
				CompleteLearningQueue();

				Debug.Assert(_modules != null);
				Debug.Assert(_x_F1_simd != null);

				var classID = UNDEFINED;
				var confidence = 1.0m;

				if(mask != null)
					Debug.Assert(_x_F0_len == mask.LongLength);

				if(_modules![ModuleNum - 1]._nodeNum >= 1) {		// net not empty

					if(nu < 1) {
						nu = 1;
						Common.Warning("Invalid value for nu, changed to " + nu);
					}

					encode();
					var maskSimd = ConvertMask(mask);

					if(_modules![ModuleNum - 1].ComputeAlternativeChoiceFunctionsWithMaskAndNu(
						_x_F1_simd!, maskSimd, nu, out Stack<FTA_F2_node> enclosingNodes, out List<FTA_F2_node> neighbouringNodes)) {
						if(enclosingNodes.Count > 0) {
							var minCategorySize = _x_F0_len * Common.ScalingFactor + (long)(0.0001m * Common.ScalingFactor);
							long enclosingNodesCount = 0;
							do									// add at least one node
							{
								var currentNode = (FTAC_F2_node)enclosingNodes.Pop();
								var tmpCategorySize = currentNode.Size;

								if(tmpCategorySize < minCategorySize) {
									minCategorySize = tmpCategorySize;
									classID = currentNode.ClassID;
								}

								++enclosingNodesCount;
							} while((enclosingNodes.Count > 0) && (enclosingNodesCount < nu));
						} else if(neighbouringNodes.Count > 0) {
							var invSum = 0.0m;
							var y_F2 = new decimal[neighbouringNodes.Count];
							for(long i = 0; i < neighbouringNodes.Count; ++i) {
								// division-by-zero is prevented by the previous branch
								y_F2[i] = Common.ScalingFactor / (decimal)(Common.ScalingFactor - neighbouringNodes[(int)i].Activation);
								invSum += y_F2[i];
							}

							// discrimination function: per-class sums in two small parallel buffers (at most one
							// distinct class per neighbouring node)
							var classIDs = new long[neighbouringNodes.Count];
							var classSums = new decimal[neighbouringNodes.Count];
							var classNum = 0;

							for(long i = 0; i < neighbouringNodes.Count; ++i) {
								y_F2[i] /= invSum;

								var currentClassID = ((FTAC_F2_node)neighbouringNodes[(int)i]).ClassID;
								var found = false;
								for(var j = 0; j < classNum; ++j) {
									if(classIDs[j] == currentClassID) {
										classSums[j] += y_F2[i];
										found = true;
										break;
									}
								}
								if(!found) {
									classIDs[classNum] = currentClassID;
									classSums[classNum] = y_F2[i];
									++classNum;
								}
							}

							confidence = (decimal)neighbouringNodes[0].Activation / Common.ScalingFactor;

							// the smallest class ID wins ties, as with the former ascending-key iteration
							var maxDiscriminationValue = 0.0m;
							for(var j = 0; j < classNum; ++j) {
								if((classSums[j] > maxDiscriminationValue) ||
								   ((classSums[j] == maxDiscriminationValue) && (classIDs[j] < classID))) {
									maxDiscriminationValue = classSums[j];
									classID = classIDs[j];
								}
							}
						}
					}
				}

				return new TopoART_C_prediction(classID, confidence);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override HeaderInfo LoadBinaryHeader(BinaryReader reader)
		{
			return Common.LoadBinaryHeader(reader, _networkType, _networkName, new FileFormatVersions(FileFormatVersion, TopoARTFileFormatVersion),
				integerType, floatType);
		}

		private protected override void LoadPrecedingBinaryInformation(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo)
		{
			base.LoadPrecedingBinaryInformation(reader, fileFormatInfo);

			if(fileFormatInfo.Item1.FileFormatVersion >= 1.0m)
				Nu = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*       TopoART-C network       *");
			writer.WriteLine("*      LibTopoART (v" + LibTopoART_info.version.ToString(CultureInfo.InvariantCulture) + ")       *");
			writer.WriteLine("*********************************");
			writer.WriteLine("file format versions: " + FileFormatVersion.ToString(CultureInfo.InvariantCulture) + "; " + TopoARTFileFormatVersion.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("integer type: " + IntegerType);
			writer.WriteLine("float type: " + FloatType);
			writer.WriteLine("*********************************");
		}

		private protected override void SavePrecedingTextInformation(TextWriter writer)
		{
			base.SavePrecedingTextInformation(writer);
			writer.WriteLine("nu: " + Nu);
			writer.WriteLine("*********************************");
		}

		private protected override void SaveBinaryHeader(BinaryWriter writer, bool compatibilityMode, CompressionLevel compression)
		{
			if(compatibilityMode)
				Common.SaveCompatibleBinaryHeader(writer, _networkType, new FileFormatVersions(FileFormatVersion, TopoARTFileFormatVersion), compression);
			else
				Common.SaveBinaryHeader(writer, _networkType, new FileFormatVersions(FileFormatVersion, TopoARTFileFormatVersion),
					(integerType, floatType), compression);
		}

		private protected override void SavePrecedingBinaryInformation(BinaryWriter writer, bool compatibilityMode)
		{
			base.SavePrecedingBinaryInformation(writer, compatibilityMode);
			writer.Write(Nu);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}
	}

//**********************************************************************************************************************

}