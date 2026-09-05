/*"*********************************************************************************************************************
*                                                Episodic TopoART class                                                *
*                                     created by Marko Tscherepanow, 4 August 2013                                     *
************************************************************************************************************************
*                            $Id: Fast_Episodic_TopoART.cs 1832 2026-07-17 06:24:27Z marko $                           *
***********************************************************************************************************************/

using System.IO;
using System.Numerics;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Fast_Episodic_TopoART</c> provides an implementation of the Episodic TopoART neural network as
	/// proposed in "Marko Tscherepanow, Sina Kühnel, and Sören Riechers (2012). Episodic Clustering of Data Streams
	/// Using a Topology-Learning Neural Network. In Proceedings of the European Conference on Artificial Intelligence
	/// (ECAI), Workshop on Active and Incremental Learning (AIL), pp. 24-29. Montpellier, France."</summary>
	public class Fast_Episodic_TopoART : Fast_TopoART_base, IFast_Episodic_TopoART
	{
		private long _t_F0;
		private long _t_max;
		private const string _networkName = "Episodic TopoART";
		private const NetworkType _networkType = NetworkType.EpisodicTopoART;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Episodic_TopoART</c>.</summary>
		public new decimal FileFormatVersion
		{
			get => Common.Episodic_TopoART_file_format_version;
		}

//----------------------------------------------------------------------------------------------------------------------

		private Fast_TopoART_module CreateEpisodicTopoARTModule(long inputLen, int rho, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
		{
			return new Fast_Episodic_TopoART_module(inputLen, rho, F2_node_create_func);
		}

		private Fast_TopoART_module LoadEpisodicTopoARTModule(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func, LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			return new Fast_Episodic_TopoART_module(reader, fileFormatInfo, F2_node_create_func, F2_node_load_func);
		}

		private FTA_F2_node CreateEpisodicTopoARTF2Node(long nodeID, long inputLen, Vector<int>[] spatialWeights,
			long[]? temporalWeights)
		{
			Debug.Assert(temporalWeights != null);
			return new FETA_F2_node(nodeID, inputLen, _t_max, spatialWeights, temporalWeights!);
		}

		private FTA_F2_node LoadEpisodicTopoARTF2Node(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo)
		{
			return new FETA_F2_node(reader, fileFormatInfo, _t_max);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <value>Property <c>T_max</c> represents the maximum considered time frame.</value>
		public long T_max
		{
			get => _t_max / Common.ScalingFactor;
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This constructor initialises an Episodic TopoART network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of Episodic TopoART modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first Episodic TopoART module (ETA a).</param>
		/// <param name="t_max">The parameter limiting the considered time frame.</param>
		public Fast_Episodic_TopoART(long inputLen, long moduleNum, decimal rho_a, long t_max)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);

			if(t_max <= 0) {
				this._t_max = 100 * Common.ScalingFactor;
				Common.Warning("Invalid value for t_max, changed to " + T_max);
			}
			else if(t_max > long.MaxValue / Common.ScalingFactor) {
				this._t_max = (long.MaxValue / Common.ScalingFactor) * Common.ScalingFactor;
				Common.Warning("Too large value for t_max, changed to " + T_max);
			}
			else
				this._t_max = t_max * Common.ScalingFactor;

			Common.Message("t^max set to " + T_max);

			InitModules(inputLen << 1, CreateEpisodicTopoARTModule, CreateEpisodicTopoARTF2Node);
		}

		/// <summary>This constructor loads a saved Episodic TopoART network.</summary>
		/// <param name="path">The path of a binary Episodic TopoART file.</param>
		/// <exception cref="InvalidFileException">Thrown when the given file cannot be loaded.</exception>
		public Fast_Episodic_TopoART(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			LoadNetwork(file);
		}

		/// <summary>This constructor loads a saved Episodic TopoART network from a stream. The stream is left open.
		/// </summary>
		/// <param name="stream">A readable <c>Stream</c> containing a network in the binary Episodic TopoART file
		/// format.</param>
		/// <exception cref="InvalidFileException">Thrown when the given stream cannot be loaded.</exception>
		public Fast_Episodic_TopoART(Stream stream)
		{
			LoadNetwork(stream);
		}

		private void LoadNetwork(Stream stream)
		{
			using var reader = LoadTopoARTParams(stream, TopoARTMatchFunction, out var headerInfo);
			InitModules(reader, headerInfo, LoadEpisodicTopoARTModule, CreateEpisodicTopoARTF2Node, LoadEpisodicTopoARTF2Node);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step.
		/// <para>The spatial weights are adapted as in the original TopoART network. In contrast, the adaptation of the
		/// temporal weight <c>w_{j,2}^{F2,t}</c> occurring only in Episodic TopoART is slightly different:
		/// <c>w_{j,2}^{F2,t}(t+1) = beta_j * Max(t_2^{F1}(t), w_{j,2}^{F2,t}(t) + (1 - beta_j) * w_{j,2}^{F2,t}(t)</c>
		/// for <c>j = bm</c> or <c>j = sbm</c>. (Note: <c>w_{j,1}^{F2,t}</c> remains constant over the lifetime of a
		/// node.)</para>
		/// </summary>
		/// <param name="input">The input vector to be learnt. The input values are internally scaled from [0, 255] to
		/// [0, 1].</param>
		public override void Learn(byte[] input)
		{
			EncodeCurrentInputSimd(input);
			LearnCommon();
		}

		/// <summary>This method performs a single training step.
		/// <para>The spatial weights are adapted as in the original TopoART network. In contrast, the adaptation of the
		/// temporal weight <c>w_{j,2}^{F2,t}</c> occurring only in Episodic TopoART is slightly different:
		/// <c>w_{j,2}^{F2,t}(t+1) = beta_j * Max(t_2^{F1}(t), w_{j,2}^{F2,t}(t) + (1 - beta_j) * w_{j,2}^{F2,t}(t)</c>
		/// for <c>j = bm</c> or <c>j = sbm</c>. (Note: <c>w_{j,1}^{F2,t}</c> remains constant over the lifetime of a
		/// node.)</para>
		/// </summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			EncodeCurrentInputSimd(input);
			LearnCommon();
		}

		private void LearnCommon()
		{
			Debug.Assert(_modules != null);
			Debug.Assert(_x_F1_simd != null);

			_t_F0 = LearningSteps++;

			// set temporal input
			long[] t_F1 = [ _t_F0 * Common.ScalingFactor, _t_F0 * Common.ScalingFactor ];

			var input = _x_F1_simd!;

			var lr = LearningResult.PropagateFurther;
			Learn(ModuleNum, m => {
					if(lr == LearningResult.PropagateFurther)
						lr = ((Fast_Episodic_TopoART_module)_modules![m]).LearnWithMask(input, t_F1, TopoARTMatchFunction, _alpha, _beta_sbm, _phis![m], _skipEdgeLearning);

					if((_modules![m].LearningCycles != 0) && ((_modules![m].LearningCycles % Tau) == 0))
						_modules![m].RemoveNodeCandidates(_phis![m]);
			});
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

			_t_F0 = reader.ReadInt64();
			_t_max = reader.ReadInt64();

			_t_max *= Common.ScalingFactor;
			Common.Message("t^max set to " + T_max);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*   Episodic TopoART network    *");
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
			writer.WriteLine("t^F0: " + _t_F0);
			writer.WriteLine("t^max: " + T_max);
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
			writer.Write(_t_F0);
			writer.Write(T_max);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method starts the recall process.</summary>
		/// <param name="stimulus">The stimulus (input) which is used to trigger recall. The stimulus elements are
		/// internally scaled from [0, 255] to [0, 1].</param>
		/// <returns>The number of F3 nodes created.</returns>
		public long BeginRecall(byte[] stimulus)
		{
			Debug.Assert(_x_F0_len == stimulus.LongLength);
			Debug.Assert(_modules != null);

			EncodeCurrentInputSimd(stimulus);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).BeginRecall(_x_F1_simd!, _phis![ModuleNum - 1]);
			}
		}

		/// <summary>This method starts the recall process.</summary>
		/// <param name="stimulus">The stimulus (input) which is used to trigger recall.</param>
		/// <returns>The number of F3 nodes created.</returns>
		public long BeginRecall(decimal[] stimulus)
		{
			Debug.Assert(_x_F0_len == stimulus.LongLength);
			Debug.Assert(_modules != null);

			EncodeCurrentInputSimd(stimulus);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).BeginRecall(_x_F1_simd!, _phis![ModuleNum - 1]);
			}
		}

		/// <summary>This method performs a single inter-episode recall step and sets the starting point for
		/// intra-episode recall.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
		public bool InterEpisodeRecallStep(out byte[]? recallResult, out decimal F3_activation)
		{
			Debug.Assert(_modules != null);
			return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).InterEpisodeRecallStep(out recallResult, out F3_activation);
		}

		/// <summary>This method performs a single inter-episode recall step and sets the starting point for
		/// intra-episode recall.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
		public bool InterEpisodeRecallStep(out decimal[]? recallResult, out decimal F3_activation)
		{
			Debug.Assert(_modules != null);
			return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).InterEpisodeRecallStep(out recallResult, out F3_activation);
		}

		/// <summary>This method performs a single intra-episode recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed or not.</returns>
		public bool IntraEpisodeRecallStep(out byte[]? recallResult)
		{
			Debug.Assert(_modules != null);
			return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).IntraEpisodeRecallStep(out recallResult);
		}

		/// <summary>This method performs a single intra-episode recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed, or not.</returns>
		public bool IntraEpisodeRecallStep(out decimal[]? recallResult)
		{
			Debug.Assert(_modules != null);
			return ((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).IntraEpisodeRecallStep(out recallResult);
		}

		/// <summary>This method stops the recall process and frees temporary resources.</summary>
		public void EndRecall()
		{
			Debug.Assert(_modules != null);
			((Fast_Episodic_TopoART_module)_modules![ModuleNum - 1]).EndRecall();
		}
	}

//**********************************************************************************************************************

}
