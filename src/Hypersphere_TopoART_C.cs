/*"*********************************************************************************************************************
*                                              Hypersphere TopoART-C class                                             *
*                                    created by Marko Tscherepanow, 8 November 2017                                    *
************************************************************************************************************************
*                            $Id: Hypersphere_TopoART_C.cs 1680 2025-11-15 17:23:15Z marko $                           *
***********************************************************************************************************************/

using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Hypersphere_TopoART_C</c> provides an implementation of the Hypersphere TopoART-C neural
	/// network. Hypersphere TopoART-C is a combination of Hypersphere TopoART as proposed in "Marko Tscherepanow
	/// (2012). Incremental On-line Clustering with a Topology-Learning Hierarchical ART Neural Network Using
	/// Hyperspherical Categories. In Poster and Industry Proceedings of the Industrial Conference on Data Mining
	/// (ICDM), pp. 22–34. Fockendorf, Germany: ibai-publishing." and TopoART-C as proposed in "Marko Tscherepanow and 
	/// Sören Riechers (2012). An Incremental On-line Classifier for Imbalanced, Incomplete, and Noisy Data. In
	/// Proceedings of the European Conference on Artificial Intelligence (ECAI), Workshop on Active and Incremental
	/// Learning (AIL), pp. 18-23. Montpellier, France."
	/// <para>In contrast to classes <c>TopoART_C</c> and <c>Fast_TopoART_C</c>, class <c>Hypersphere_TopoART_C</c> does
	/// not require all input to lie in the interval [0, 1]. The input range is controlled by the radial extend
	/// parameter R.</para>
	/// </summary>
	public class Hypersphere_TopoART_C : Hypersphere_TopoART, ITopoART_C
	{
		/// <summary>Instance variable <c>UNDEFINED</c> gives the value used for indicating undefined and uninitialised
		/// variables.</summary>
		private const long UNDEFINED = LibTopoART_info.UNDEFINED;

		/// <summary>Instance variable <c>UNDEFINED_CLASS_ID</c> gives the value used for indicating that an input
		/// sample was predict to belong to the undefined class; i.e, no class ID was provided for such input samples
		/// during training.</summary>
		public const long UNDEFINED_CLASS_ID = -2;

		private const string _networkName = "Hypersphere TopoART-C";
		private const NetworkType _networkType = NetworkType.HypersphereTopoARTC;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Hypersphere_TopoART_C</c>.</summary>
		public new decimal FileFormatVersion { get => Common.Hypersphere_TopoART_C_file_format_version; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the set of
		/// enclosing categories E and the neighbourhood set N during prediction. If the parameter <c>nu</c> is not
		/// explicitly provided for prediction, this property will be applied. (This parameter does not modify the
		/// network. It may be arbitrarily changed for each prediction step.)</summary>
		public long Nu
		{
			get => field;
			set {
				field = value;
				if(field < 1) {
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
				_skipEdgeLearning = value;
				Common.Message((value ? "Disable" : "Enable") + " edge learning");
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This constructor initialises a Hypersphere TopoART-C network and sets the radial extend parameter
		/// to <c>Math.Sqrt(inputLen)/2</c>.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of Hypersphere TopoART-C modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first Hypersphere TopoART-C module (HTA-C a).</param>
		public Hypersphere_TopoART_C(long inputLen, long moduleNum, decimal rho_a) : 
			this(inputLen, moduleNum, rho_a, (decimal)Math.Sqrt(inputLen) / 2.0m) {}

		/// <summary>This constructor initialises a Hypersphere TopoART-C network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of Hypersphere TopoART-C modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first Hypersphere TopoART-C module (HTA-C a).</param>
		/// <param name="R">The radial extend parameter.</param>
		public Hypersphere_TopoART_C(long inputLen, long moduleNum, decimal rho_a, decimal R)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);

			if(R <= 0.0m) {
				this.R = (decimal)Math.Sqrt(inputLen) / 2.0m;
				Common.Warning("Too small value for R, changed to " + this.R);
			} else
				this.R = R;

			Common.Message($"R set to {this.R:0.##########}");

			InitModules(inputLen + 1, CreateTopoARTModule, null);
		}

		/// <summary>This constructor loads a saved Hypersphere TopoART-C network.</summary>
		/// <param name="path">The path of a binary Hypersphere TopoART-C file.</param>
		/// <exception cref="InvalidFileException">Throws when the given file cannot be loaded.</exception>
		public Hypersphere_TopoART_C(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read); 

			var fileFormatVersions = LoadTopoARTParams(file, null, out var outerReader);
			InitModules(outerReader, fileFormatVersions, LoadTopoARTModule, null, (BinaryReader reader, 
				in (FileFormatVersions localFileFormatVersions, bool) localFileFormatInfo) => 
					new HTAC_F2_node(reader, localFileFormatInfo.localFileFormatVersions, R));
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
		/// <exception cref="InvalidClassIDException">Throws when <paramref name="classID"/> is less than 0.</exception>
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
					return(new HTAC_F2_node(nodeID, inputLen, spatialWeights, R, classID));
				};

			Debug.Assert(_modules != null);

			MatchFunction<TA_F2_node, decimal> matchFunction = (node, rho) => 
				(node.MatchValue >= rho) && (((HTAC_F2_node)node).ClassID == classID);

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
			TopoART_C_prediction pred = Predict(input, null, nu);
			return pred.classID;
		}

		/// <summary>This method predicts the class ID using the default value of nu.</summary>
		/// <param name="input">The input vector the class ID of which is to be predicted.</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns> An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
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
		/// <returns> An object of type <c>TopoART_C_prediction</c> containing the predicted class ID and a
		/// corresponding confidence value.</returns>
		public TopoART_C_prediction Predict(decimal[] input, bool[]? mask, long nu)
		{
			lock(_learningLock) {
				CompleteLearningQueue();
				
				decimal[] x_F1;
				var classID = UNDEFINED;
				var confidence = 1.0m;

				Debug.Assert(_x_F0_len == input.LongLength);
				if(mask != null)
					Debug.Assert(_x_F0_len == mask.LongLength);

				Debug.Assert(_modules != null);
				Debug.Assert(_x_F0 != null);

				if(_modules![ModuleNum - 1]._nodeNum >= 1) {		// net not empty
					if(nu == 0) {
						nu = 1;
						Common.Warning("Invalid value for nu, changed to " + nu);
					}

					for(long i = 0; i < _x_F0_len; ++i)
						_x_F0![i] = input[i];

					x_F1 = EncodeCurrentInput();
					
					if(_modules![ModuleNum - 1].ComputeAlternativeChoiceFunctionsWithMaskAndNu(
						x_F1, mask, nu, out Stack<TA_F2_node> enclosingNodes, out List<TA_F2_node> neighbouringNodes)) {
						if(enclosingNodes.Count > 0) {
							var minCategorySize = _x_F0_len + 0.0001m;
							long enclosingNodesCount = 0;
							do									// add at least one node
							{
								var currentNode = (HTAC_F2_node)enclosingNodes.Pop();
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

							var discriminationFunction = new SortedDictionary<long, decimal>();

							for(long i = 0; i < neighbouringNodes.Count; ++i) {
								y_F2[i] /= invSum;
								if(discriminationFunction.ContainsKey(((HTAC_F2_node)neighbouringNodes[(int)i]).ClassID))
									discriminationFunction[((HTAC_F2_node)neighbouringNodes[(int)i]).ClassID] += y_F2[i];
								else
									discriminationFunction.Add(((HTAC_F2_node)neighbouringNodes[(int)i]).ClassID, y_F2[i]); 
							}

							confidence = neighbouringNodes[0].Activation;

							var maxDiscriminationValue = 0.0m;
							foreach(KeyValuePair<long, decimal> pair in discriminationFunction) {
								if(pair.Value > maxDiscriminationValue) {
									maxDiscriminationValue = pair.Value;
									classID = pair.Key;
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
			var headerInfo = Common.LoadBinaryHeader(reader, _networkType, _networkName, new FileFormatVersions(FileFormatVersion,
								HypersphereTopoARTFileFormatVersion, TopoARTFileFormatVersion), integerType,
								floatType);
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
			writer.WriteLine("* Hypersphere TopoART-C network *");
			writer.WriteLine("*      LibTopoART (v" + LibTopoART_info.version.ToString(CultureInfo.InvariantCulture) + ")       *");
			writer.WriteLine("*********************************");
			writer.WriteLine("file format versions: " + FileFormatVersion.ToString(CultureInfo.InvariantCulture)
							 + "; " + HypersphereTopoARTFileFormatVersion.ToString(CultureInfo.InvariantCulture)
							 + "; " + TopoARTFileFormatVersion.ToString(CultureInfo.InvariantCulture));
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
			Common.SaveBinaryHeader(writer, _networkType, new FileFormatVersions(FileFormatVersion, HypersphereTopoARTFileFormatVersion, TopoARTFileFormatVersion),
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
			Common.InitialisationMessage(_networkName);
		}
	}

//**********************************************************************************************************************

}
