/*"*********************************************************************************************************************
*                                                    TopoART class                                                     *
*                                      created by Marko Tscherepanow, 12 June 2011                                     *
************************************************************************************************************************
*                                   $Id: TopoART.cs 1845 2026-08-15 14:52:37Z marko $                                  *
***********************************************************************************************************************/

using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using System.Collections.Generic;
using System.Text;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>TopoART</c> provides an implementation of the TopoART neural network as proposed in
	/// "Marko Tscherepanow (2010). TopoART: A topology learning hierarchical ART network. In Proceedings of the
	/// International Conference on Artificial Neural Networks (ICANN), LNCS 6354, pp. 157–167. Berlin, Germany:
	/// Springer."
	/// <para>Internally, real-valued data are stored in <c>decimal</c> variables. Hence,
	/// computations are rather slow but very accurate.</para>
	/// <para>Class <c>TopoART</c> requires all input to lie in the interval [0, 1].</para>
	/// </summary>
	public class TopoART : Network_base, ITopoART, ITopoART_base_stream, ICategoryAccess, IDisposable
	{
		/// <summary>Instance variable <c>x_F0</c> represents the current input vector.</summary>
		private protected decimal[]? _x_F0;

		private protected TopoART_module[]? _modules;

		private protected const TypeIndex integerType = TypeIndex.LongIndex;
		private protected const TypeIndex floatType = TypeIndex.DecimalIndex;

		private const string _networkName = "TopoART";
		private const NetworkType _networkType = NetworkType.TopoART;

		private bool _disposed;

//----------------------------------------------------------------------------------------------------------------------

		private const long _serialPerNodeWorkOffset = 6;

		private protected TopoART_module CreateTopoARTModule(long inputLen, decimal rho, CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func)
		{
			return new TopoART_module(inputLen, rho, F2_node_create_func, _serialPerNodeWorkOffset);
		}

		private protected TopoART_module LoadTopoARTModule(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func, LoadF2Node<TA_F2_node> F2_node_load_func)
		{
			Debug.Assert(fileFormatInfo.Item2 == false);
			return new TopoART_module(reader, fileFormatInfo.Item1, F2_node_create_func, F2_node_load_func, _serialPerNodeWorkOffset);
		}

		private TA_F2_node CreateTopoARTF2Node(long nodeID, long inputLen, decimal[] spatialWeights,
		    long[]? temporalWeights)
		{
			Debug.Assert(temporalWeights == null);
			return new TA_F2_node(nodeID, inputLen, spatialWeights);
		}

		private TA_F2_node LoadTopoARTF2Node(BinaryReader reader, in (FileFormatVersions fileFormatVersions, bool) fileFormatInfo)
		{
			return new TA_F2_node(reader, fileFormatInfo.fileFormatVersions);
		}

		private protected static bool TopoARTMatchFunction(TA_F2_node node, decimal rho)
		{
			return node.MatchValue >= rho;
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <value>Property <c>Alpha</c> represents the choice parameter alpha.</value>
		public decimal Alpha
		{
			get => field;
			set {
				if(LearningSteps == 0) {
					if(value <= 0.0m) {
						field = 0.001m;
						Common.Warning("Too small value for alpha, changed to " + Alpha);
					} else {
						field = value;
						if (value > 0.1m)
							Common.Warning("alpha might be too large");
					}
					Common.Message($"alpha set to {Alpha:0.##########}");
				} else
					Common.Warning("Unable to set alpha after training started, keep old value " + Alpha);
			}
		}

		/// <value>Property <c>Beta_sbm</c> represents the learning rate of the second best-matching nodes.</value>
		public decimal Beta_sbm
		{
			get => field;
			set {
				if(LearningSteps == 0) {
					if(value < 0.0m) {
						field = 0.0m;
						Common.Warning("Too small value for beta_sbm, changed to " + Beta_sbm);
					} else if (value > 1.0m) {
						field = 1.0m;
						Common.Warning("Too large value for beta_sbm, changed to " + Beta_sbm);
					} else
						field = value;
					Common.Message($"beta_sbm set to {Beta_sbm:0.##########}");
				} else
					Common.Warning("Unable to set beta_sbm after training started, keep old value " + Beta_sbm);
			}
		}

		/// <value>Property <c>ClusterNum</c> represents the number of TopoART clusters found by each module.</value>
		public long[] ClusterNum
		{
			get {
				Debug.Assert(_modules != null);

				var resultArray = new long[ModuleNum];

				lock(_learningLock) {
					CompleteLearningQueue();

					for(long i = 0; i < ModuleNum; ++i)
						resultArray[i] = _modules![i]._clusterNum;
				}

				return resultArray;
			}
		}

		/// <value>Property <c>NodeNum</c> represents the number of TopoART nodes used by each module.</value>
		public long[] NodeNum
		{
			get {
				Debug.Assert(_modules != null);

				var resultArray = new long[ModuleNum];

				lock(_learningLock) {
					CompleteLearningQueue();

					for(long i = 0; i < ModuleNum; ++i)
						resultArray[i] = _modules![i]._nodeNum;
				}

				return resultArray;
			}
		}

		/// <value>Property <c>Rho_a</c> represents the vigilance parameter of the first TopoART module (TA a).</value>
		public decimal Rho_a { get; private set; }

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>IntegerType</c> returns a string containing the data type used for representing integer
		/// variables (IDs, parameters, counters, etc.) internally.</summary>
		public string IntegerType { get; } = Common.Types[(int)integerType];

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>TopoART</c>.</summary>
		public decimal FileFormatVersion { get => Common.TopoART_file_format_version; }

		/// <summary>Property <c>FloatType</c> returns a string containing the data type used for representing floating
		/// point variables (input, weights, etc.) internally.</summary>
		public string FloatType { get; } = Common.Types[(int)floatType];

		/// <summary>Property <c>TopoARTFileFormatVersion</c> returns the version of the file format used by class
		/// <c>TopoART</c>.</summary>
		public decimal TopoARTFileFormatVersion { get => FileFormatVersion; }

//----------------------------------------------------------------------------------------------------------------------

		// Do not use!
		private protected TopoART() {}

		/// <summary>This constructor initialises a TopoART network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART module (TA a).</param>
		public TopoART(long inputLen, long moduleNum, decimal rho_a)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);
			InitModules(inputLen << 1, CreateTopoARTModule, CreateTopoARTF2Node);
		}

		private protected void SetTopoARTParams(long inputLen, long moduleNum, decimal rho_a)
		{
			InitialisationMessage();

			if(inputLen < 1) {
				inputLen = 1;				// check required, as otherwise HTA causes a division by zero
				Common.Warning("Invalid length of the input vector x_F0, changed to " + inputLen);
			}
			_x_F0_len = inputLen;

			_x_F0 = new decimal[_x_F0_len];
			for(long i = 0; i < _x_F0_len; ++i)
				_x_F0[i] = 0.0m;

			if(rho_a < 0.0m) {
				Rho_a = 0.0m;
				Common.Warning("Too small value for rho_a, changed to " + Rho_a);
			} else if (rho_a > 1.0m) {
				Rho_a = 1.0m;
				Common.Warning("Too large value for rho_a, changed to " + Rho_a);
			} else
				Rho_a = rho_a;
			Common.Message($"rho_a set to {Rho_a:0.##########}");

			Beta_sbm = 0.5m;
			Tau = 100;
			LearningSteps = 0;
			Alpha = 0.001m;

			if(moduleNum < 1) {
				ModuleNum = 1;
				Common.Warning("Invalid number of modules, changed to " + ModuleNum);
			} else
				ModuleNum = moduleNum;

			_phis = new long[ModuleNum];
			Phi = 1;
		}

		/// <summary>This constructor loads a saved TopoART network.</summary>
		/// <param name="path">The path of a binary TopoART file.</param>
		/// <exception cref="InvalidFileException">Thrown when the given file cannot be loaded.</exception>
		public TopoART(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			LoadNetwork(file);
		}

		/// <summary>This constructor loads a saved TopoART network from a stream. The stream is left open.</summary>
		/// <param name="stream">A readable <c>Stream</c> containing a network in the binary TopoART file format.</param>
		/// <exception cref="InvalidFileException">Thrown when the given stream cannot be loaded.</exception>
		public TopoART(Stream stream)
		{
			LoadNetwork(stream);
		}

		private void LoadNetwork(Stream stream)
		{
			using var reader = LoadTopoARTParams(stream, TopoARTMatchFunction, out var fileFormatVersions);
			InitModules(reader, fileFormatVersions, LoadTopoARTModule, CreateTopoARTF2Node, LoadTopoARTF2Node);
		}

		private protected BinaryReader LoadTopoARTParams(Stream stream, MatchFunction<TA_F2_node, decimal>? initialMatchFunction,
			out FileFormatVersions fileFormatVersions)
		{
			(FileFormatVersions fileFormatVersions, SaveFlags flags) headerInfo;
			using(var headerReader = new BinaryReader(stream, Encoding.UTF8, true))
				headerInfo = LoadBinaryHeader(headerReader);

			fileFormatVersions = headerInfo.fileFormatVersions;

			var reader = headerInfo.flags == SaveFlags.None ? new BinaryReader(stream, Encoding.UTF8, true) : new BinaryReader(new GZipStream(stream, CompressionMode.Decompress, true));

			try {
				LoadPrecedingBinaryInformation(reader, headerInfo.fileFormatVersions);

				_x_F0_len = reader.ReadInt64();
				_x_F0 = new decimal[_x_F0_len];
				for(long i = 0; i < _x_F0_len; ++i)
					_x_F0[i]=reader.ReadDecimal();

				var tmpRhoA = reader.ReadDecimal();
				if(tmpRhoA < 0.0m) {
					Rho_a = 0.0m;
					Common.Warning("Too small value for rho_a, changed to " + Rho_a);
				} else if (tmpRhoA > 1.0m) {
					Rho_a = 1.0m;
					Common.Warning("Too large value for rho_a, changed to " + Rho_a);
				} else
					Rho_a = tmpRhoA;

				Common.Message($"rho_a set to {Rho_a:0.##########}");

				Beta_sbm = reader.ReadDecimal();
				Tau = reader.ReadInt64();
				var tmpPhi = reader.ReadInt64();
				var tmpLearningSteps = reader.ReadInt64();
				Alpha = reader.ReadDecimal();

				_skipEdgeLearning = headerInfo.fileFormatVersions.TopoARTFileFormatVersion >= 1.0m ? reader.ReadBoolean() : false;

				ModuleNum = reader.ReadInt64();
				_phis = new long[ModuleNum];

				// load phi array if required
				if(tmpPhi == LibTopoART_info.UNDEFINED)
					for(long i = 0; i < ModuleNum; ++i)
						_phis[i] = reader.ReadInt64();
				else
					Phi = tmpPhi;

				LearningSteps = tmpLearningSteps;

				return reader;
			} catch {
				reader.Dispose();
				throw;
			}
		}

		private protected virtual (FileFormatVersions, SaveFlags) LoadBinaryHeader(BinaryReader reader)
		{
			var headerInfo = Common.LoadBinaryHeader(reader, _networkType, _networkName, FileFormatVersion, integerType, floatType);
			return (headerInfo.FileFormatVersions, headerInfo.Flags);
		}

		private protected virtual void LoadPrecedingBinaryInformation(BinaryReader reader, FileFormatVersions fileFormatVersions) {}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Releases unmanaged resources and performs other cleanup operations before the
		/// <see cref="LibTopoART.TopoART"/> is reclaimed by garbage collection.</summary>
		~TopoART()
		{
			Dispose(false);
		}

		/// <summary>Releases all resources used by the <see cref="LibTopoART.TopoART"/> object.</summary>
		/// <remarks>Call <see cref="Dispose()"/> when you are finished using the <see cref="LibTopoART.TopoART"/>. The
		/// <see cref="Dispose()"/> method leaves the <see cref="LibTopoART.TopoART"/> in an unusable state. After
		/// calling <see cref="Dispose()"/>, you must release all references to the <see cref="LibTopoART.TopoART"/> so
		/// the garbage collector can reclaim the memory that the <see cref="LibTopoART.TopoART"/> was occupying.
		/// </remarks>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>Release resources used by the <see cref="LibTopoART.TopoART"/> object.</summary>
		/// <param name="disposing">If set to <c>true</c> all managed resources are released.</param>
		protected virtual void Dispose(bool disposing)
		{
			if(!_disposed) {
				if(disposing) {
					if(_modules != null) {
						lock(_learningLock) {
							CompleteLearningQueueNoThrow();

							for(long i = 0; i < ModuleNum; ++i) {
								if(_modules[i] != null) {
									_modules[i].Dispose();
									_modules[i] = null!;
								}
							}
						}
					}
					_modules = null;
				}
				_disposed = true;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method computes the cluster IDs for all neurons.</summary>
		public void ComputeClusterIDs()
		{
			Debug.Assert(_modules != null);

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i)
					_modules![i].ComputeClusterIDs(_phis![i]);
			}
		}

		private protected virtual decimal[] EncodeCurrentInput()
		{
			Debug.Assert(_x_F0 != null);

			var adaptedInput = false;

			var x_F1 = new decimal[_x_F0_len << 1];
			for(long i = 0; i < _x_F0_len; ++i) {
				var element = _x_F0![i];
				if(element < 0.0m) {
					element = 0.0m;
					adaptedInput = true;
				} else if(element > 1.0m) {
					element = 1.0m;
					adaptedInput = true;
				}
				x_F1[i] = element;
				x_F1[i + _x_F0_len] = 1.0m - element;
			}

			if(adaptedInput)
				Common.Warning(Common.Warning_CorrectedInputValue, VerbosityLevel.Verbose);

			return x_F1;
		}

		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t).</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		public F2_output[] GetBMOutput(decimal[] input)
		{
			return GetBMOutput(input, null);
		}

		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t).</param>
		/// <param name="mask">A mask vector excluding individual dimensions of x(t) from the computation. (Setting an
		/// element of the mask vector to <c>true</c>, excludes the corresponding elements of x(t).)</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		public F2_output[] GetBMOutput(decimal[] input, bool[]? mask)
		{
			Debug.Assert(_x_F0_len == input.LongLength);
			Debug.Assert(_modules != null);
			Debug.Assert(_x_F0 != null);

			var result = new F2_output[ModuleNum];

			for(long i = 0; i < _x_F0_len; ++i)
				_x_F0![i] = input[i];

			var x_F1 = EncodeCurrentInput();

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i)
					result[i] = _modules![i].GetBMOutputWithMask(x_F1, mask, _phis![i]);
			}

			return result;
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public virtual void Learn(decimal[] input)
		{
			LearnWithMask(input, null);
		}

		private protected void LearnWithMask(decimal[] input, bool[]? mask, CreateF2Node<TA_F2_node, decimal, long>? createFunction = null,
			MatchFunction<TA_F2_node, decimal>? matchFunction = null)
		{
			Debug.Assert(_x_F0 != null);
			Debug.Assert(_modules != null);

			++LearningSteps;

			for(long i = 0; i < _x_F0_len; ++i)
				_x_F0![i] = input[i];

			var x_F1 = EncodeCurrentInput();

			var lr = LearningResult.PropagateFurther;
			Learn(ModuleNum, m => {
					if(lr == LearningResult.PropagateFurther) {
						if(createFunction != null)
							_modules![m].CreateF2NodeFunction = createFunction;
						lr = _modules![m].LearnWithMask(x_F1, mask, matchFunction ?? TopoARTMatchFunction, Alpha, Beta_sbm, _phis![m], _skipEdgeLearning);
					}

					if((_modules![m].LearningCycles != 0) && ((_modules![m].LearningCycles % Tau) == 0))
						_modules![m].RemoveNodeCandidates(_phis![m]);
			});
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method saves the entire network as a text file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		public void SaveText(string path)
		{
			using var writer = new StreamWriter(File.Open(path, FileMode.Create));
			SaveText(writer);
		}

		/// <summary>This method saves the entire network as text to a writer. The writer is flushed but left open.
		/// </summary>
		/// <param name="writer">A <c>TextWriter</c> the network is saved to.</param>
		public void SaveText(TextWriter writer)
		{
			Debug.Assert(_x_F0 != null);
			Debug.Assert(_modules != null);

			SaveTextHeader(writer);
			SavePrecedingTextInformation(writer);

			writer.WriteLine("x^F0 length: " + _x_F0_len);
			writer.Write("x^F0:");
			for(long i = 0; i < _x_F0_len; ++i)
				writer.Write(" " + _x_F0![i].ToString(CultureInfo.InvariantCulture));

			writer.Write("\n");

			writer.WriteLine("rho_a: " + Rho_a.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("beta_sbm: " + Beta_sbm.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("tau: " + Tau);
			writer.WriteLine("phi: " + Phi);
			writer.WriteLine("learning steps: " + LearningSteps);
			writer.WriteLine("alpha: " + Alpha.ToString(CultureInfo.InvariantCulture));

			if(Common.TopoART_file_format_version >= 1.0m)
				writer.WriteLine("skip edge learning: " + (_skipEdgeLearning ? "true" : "false"));

			writer.WriteLine("module number: " + ModuleNum);

			// save phi array if the values differ
			if(Phi == LibTopoART_info.UNDEFINED) {
				writer.Write("phis:");
				for(long i = 0; i < ModuleNum; ++i)
					writer.Write(" " + _phis![i]);
				writer.Write("\n");
			}

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i) {
					writer.WriteLine("*********************************");
					writer.WriteLine("*            module " + i + "           *");
					writer.WriteLine("*********************************");
					_modules![i].SaveText(writer);
				}
			}

			writer.Flush();
		}

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*        TopoART network        *");
			writer.WriteLine("*      LibTopoART (v" + LibTopoART_info.version.ToString(CultureInfo.InvariantCulture)+")       *");
			writer.WriteLine("*********************************");
			writer.WriteLine("file format version: " + FileFormatVersion.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("integer type: " + IntegerType);
			writer.WriteLine("float type: " + FloatType);
			writer.WriteLine("*********************************");
		}

		private protected override void SavePrecedingTextInformation(TextWriter writer) {}

		/// <summary>This method saves the entire network as a binary file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		/// <param name="compression">Compression level of the save file (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(string path, CompressionLevel compression = CompressionLevel.Fastest)
		{
			using var file = File.Open(path, FileMode.Create);
			Save(file, compression);
		}

		/// <summary>This method saves the entire network to a stream using the binary file format. The stream is left
		/// open.</summary>
		/// <param name="stream">A writable <c>Stream</c> the network is saved to.</param>
		/// <param name="compression">Compression level of the saved data (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(Stream stream, CompressionLevel compression = CompressionLevel.Fastest)
		{
			Debug.Assert(_x_F0 != null);
			Debug.Assert(_modules != null);

			if(TopoARTFileFormatVersion <= 0.10m)
				compression = CompressionLevel.NoCompression;

			using(var headerWriter = new BinaryWriter(stream, Encoding.UTF8, true))
				SaveBinaryHeader(headerWriter, compression);

			using var writer = compression == CompressionLevel.NoCompression ? new BinaryWriter(stream, Encoding.UTF8, true) : new BinaryWriter(new GZipStream(stream, compression, true));

			SavePrecedingBinaryInformation(writer);

			writer.Write(_x_F0_len);
			for(long i = 0; i < _x_F0_len; ++i)
				writer.Write(_x_F0![i]);

			writer.Write(Rho_a);
			writer.Write(Beta_sbm);
			writer.Write(Tau);
			writer.Write(Phi);
			writer.Write(LearningSteps);
			writer.Write(Alpha);

			if(Common.TopoART_file_format_version >= 1.0m)
				writer.Write(_skipEdgeLearning);

			writer.Write(ModuleNum);

			// save phi array if the values differ
			if(Phi == LibTopoART_info.UNDEFINED)
				for(long i = 0; i < ModuleNum; ++i)
					writer.Write(_phis![i]);

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i)
					_modules![i].Save(writer);
			}
		}

		private protected virtual void SaveBinaryHeader(BinaryWriter writer, CompressionLevel compression)
		{
			Common.SaveBinaryHeader(writer, _networkType, FileFormatVersion, (integerType, floatType), compression);
		}

		private protected virtual void SavePrecedingBinaryInformation(BinaryWriter writer) {}

//----------------------------------------------------------------------------------------------------------------------

		private protected void InitModules(long inputLen, CreateModule<TopoART_module, TA_F2_node, decimal, decimal, long> moduleCreateFunction,
								  CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func)
		{
			_modules = new TopoART_module[ModuleNum];

			long i;
			decimal rho;
			for(i = 0, rho = Rho_a; i < ModuleNum; ++i, rho = 0.5m * (rho + 1.0m)) {
				_modules[i] = moduleCreateFunction(inputLen, rho, F2_node_create_func);
			}
		}

		private protected void InitModules(BinaryReader reader, in FileFormatVersions fileFormatVersions,
								  LoadModule<TopoART_module, TA_F2_node, decimal, long> moduleLoadFunction,
								  CreateF2Node<TA_F2_node, decimal, long>? F2_node_create_func,
								  LoadF2Node<TA_F2_node> F2_node_load_func)
		{
			_modules = new TopoART_module[ModuleNum];
			for(long i = 0; i < ModuleNum; ++i)
				_modules[i] = moduleLoadFunction(reader, (fileFormatVersions, false), F2_node_create_func, F2_node_load_func);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected virtual void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method resets the adaptation state to <c>AdaptationState.NO_ADAPTATION</c>.</summary>
		/// <exception cref="InvalidNumberException">Thrown when the number of edges of an F2 node is greater than
		/// <c>int.MaxValue</c>.</exception>
		public void ResetAdaptationState()
		{
			ResetAdaptationState<TopoART_module, decimal>(_modules);
		}

		/// <summary>This method returns the current adaptation state.</summary>
		/// <param name="epsilon">The threshold for weight adaptations to be considered.</param>
		/// <returns>An enumeration describing the adaptation state.</returns>
		/// <exception cref="InvalidStateException">Thrown when the network is in an invalid state.</exception>
		/// <exception cref="InvalidNumberException">Thrown when the number of edges of an F2 node is greater than
		/// <c>int.MaxValue</c>.</exception>
		public AdaptationState GetAdaptationState(decimal epsilon = 0.001m)
		{
			Debug.Assert(_modules != null);

			lock(_learningLock) {
				CompleteLearningQueue();

				AdaptationState result = AdaptationState.NO_ADAPTATION;

				for(long i = 0; i < ModuleNum; ++i) {
					result |= _modules![i].GetAdaptationState(_phis![i], (weight1, weight2) =>  {
						var diff = weight2 - weight1;
						if(diff < 0)
							diff = -diff;
						return diff > epsilon;
					});
				}

				return result;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method checks a provided module index and sets <c>FINAL_MODULE</c> to <c>ModuleNum - 1</c>.
		/// </summary>
		/// <param name="moduleIndex">The module index to be checked.</param>
		/// <exception cref="InvalidModuleIndexException">Thrown when
		/// <paramref name="moduleIndex"/> is invalid.</exception>
		private long GetModuleIndex(long moduleIndex)
		{
			var selectedModuleIndex = (moduleIndex == FINAL_MODULE) ? ModuleNum - 1 : moduleIndex;

			if(selectedModuleIndex < 0 || selectedModuleIndex >= ModuleNum)
				throw new InvalidModuleIndexException(Common.InvalidModuleIndexException_OutOfRangeID);

			return selectedModuleIndex;
		}

		/// <summary>This method collects information on the categories of a specified module.</summary>
		/// <param name="moduleIndex">The index of the module the categories of which are to be analysed.</param>
		/// <returns>A list containing information about the respective categories.</returns>
		/// <exception cref="InvalidModuleIndexException">Thrown when <paramref name="moduleIndex"/> is invalid.
		/// </exception>
		/// <exception cref="InvalidNumberException">Thrown when the number of nodes of a module is greater than
		/// <c>int.MaxValue</c>.</exception>
		public List<CategoryInfo>? GetCategories(long moduleIndex = FINAL_MODULE)
		{
			Debug.Assert(_modules != null);

			var selectedModuleIndex = GetModuleIndex(moduleIndex);

			lock(_learningLock) {
				CompleteLearningQueue();
				return _modules![selectedModuleIndex].Categories;
			}
		}
	}

//**********************************************************************************************************************

}