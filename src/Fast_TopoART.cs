/*"*********************************************************************************************************************
*                                                    TopoART class                                                     *
*                                      created by Marko Tscherepanow, 12 June 2011                                     *
************************************************************************************************************************
*                                $Id: Fast_TopoART.cs 1604 2024-12-21 12:54:31Z marko $                                *
***********************************************************************************************************************/

using System.Numerics;
using System.IO;
using System.Diagnostics;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Fast_TopoART</c> provides an implementation of the TopoART neural network as proposed in
	/// "Marko Tscherepanow (2010). TopoART: A topology learning hierarchical ART network. In Proceedings of the
	/// International Conference on Artificial Neural Networks (ICANN), LNCS 6354, pp. 157–167. Berlin, Germany:
	/// Springer."
	/// <para>Internally, real-valued data are mapped to <c>int</c> variables. Therefore, computations are accelerated
	/// but less accurate. As a consequence, the results may differ slightly from class <c>TopoART</c>.</para>
	/// <para>Class <c>Fast_TopoART</c> requires all input to lie in the interval [0, 1].</para>
	/// </summary>
	public class Fast_TopoART : Fast_TopoART_base 
	{
		private const string _networkName = "TopoART";
		private const NetworkType _networkType = NetworkType.TopoART;

//----------------------------------------------------------------------------------------------------------------------

		private protected Fast_TopoART_module CreateTopoARTModule(long inputLen, int rho, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
		{
			return new Fast_TopoART_module(inputLen, rho, F2_node_create_func);
		}

		private protected Fast_TopoART_module LoadTopoARTModule(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func, LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			return new Fast_TopoART_module(reader, fileFormatInfo, F2_node_create_func, F2_node_load_func);
		}

		private protected FTA_F2_node CreateTopoARTF2Node(long nodeID, long inputLen, Vector<int>[] spatialWeights, 
			long[]? temporalWeights)
		{
			Debug.Assert(temporalWeights == null);
			return new FTA_F2_node(nodeID, inputLen, spatialWeights);
		}

		private protected FTA_F2_node LoadTopoARTF2Node(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo)
		{
			return new FTA_F2_node(reader, fileFormatInfo);
		}

//----------------------------------------------------------------------------------------------------------------------

		// Do not use!
		private protected Fast_TopoART() {}

		/// <summary>This constructor initialises a TopoART network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART module (TA a).</param>
		public Fast_TopoART(long inputLen, long moduleNum, decimal rho_a)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);
			InitModules(inputLen << 1, CreateTopoARTModule, CreateTopoARTF2Node);
		}

		/// <summary>This constructor loads a saved TopoART network.</summary>
		/// <param name="path">The path of a binary TopoART file.</param>
		/// <exception cref="InvalidFileException">Throws when the given file cannot be loaded.</exception>
		public Fast_TopoART(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);

			var headerInfo = LoadTopoARTParams(file, TopoARTMatchFunction, out var reader);
			InitModules(reader, headerInfo, LoadTopoARTModule, CreateTopoARTF2Node, LoadTopoARTF2Node);
			reader.Dispose();
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt. The input values are internally scaled from [0, 255] to
		/// [0, 1].</param>
		public override void Learn(byte[] input)
		{
			LearnWithMask(input, null);
		}

		private protected void LearnWithMask(byte[] input, Vector<int>[]? mask, CreateF2Node<FTA_F2_node, Vector<int>, long>? createFunction = null,
				MatchFunction<FTA_F2_node, long>? matchFunction = null)
		{
			EncodeCurrentInputSimd(input);
			LearnWithMaskCommon(mask, createFunction, matchFunction);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			LearnWithMask(input, null);
		}

		private protected void LearnWithMask(decimal[] input, Vector<int>[]? mask, CreateF2Node<FTA_F2_node, Vector<int>, long>? createFunction = null,
			MatchFunction<FTA_F2_node, long>? matchFunction = null)
		{
			EncodeCurrentInputSimd(input);
			LearnWithMaskCommon(mask, createFunction, matchFunction);
		}

		private protected void LearnWithMaskCommon(Vector<int>[]? mask, CreateF2Node<FTA_F2_node, Vector<int>, long>? createFunction = null,
			MatchFunction<FTA_F2_node, long>? matchFunction = null)
		{
			Debug.Assert(_modules != null);
			Debug.Assert(_x_F1_simd != null);

			++LearningSteps;

			var input = _x_F1_simd!;

			LearningResult lr = LearningResult.PropagateFurther;
			Learn(ModuleNum, m => {
					if(lr == LearningResult.PropagateFurther) {
						if(createFunction != null)
							_modules![m].CreateF2NodeFunction = createFunction;
						lr = _modules![m].LearnWithMask(input, mask, matchFunction ?? TopoARTMatchFunction, _alpha, _beta_sbm, Phis[m], _skipEdgeLearning);
					}

					if((_modules![m].LearningCycles % Tau) == 0)
						_modules![m].RemoveNodeCandidates(Phis[m]);
			});
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override HeaderInfo LoadBinaryHeader(BinaryReader reader)
		{
			return Common.LoadBinaryHeader(reader, _networkType, _networkName, FileFormatVersion, integerType, floatType);
		}

		private protected override void SaveBinaryHeader(BinaryWriter writer, bool compatibilityMode, CompressionLevel compression)
		{
			if(compatibilityMode)
				Common.SaveCompatibleBinaryHeader(writer, _networkType, FileFormatVersion, compression);
			else
				Common.SaveBinaryHeader(writer, _networkType, FileFormatVersion, (integerType, floatType), compression);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}
	}

//**********************************************************************************************************************

}

