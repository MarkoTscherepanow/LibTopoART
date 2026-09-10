using System;
using System.IO;
using System.Collections.Generic;
using System.Numerics;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Base class providing functionality common to several TopoART networks.</summary>
	public abstract class Fast_TopoART_base : Network_base, IFast_TopoART, ITopoART_base_stream, ICategoryAccess,
		IDisposable
	{

//----------------------------------------------------------------------------------------------------------------------

		private enum InputCheckResult
		{
			OK,
			TooLow,
			TooHigh
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Instance variable <c>x_F1_simd</c> represents the current encoded input vector.</summary>
		private protected Vector<int>[]? _x_F1_simd;

		private protected Fast_TopoART_module[]? _modules;

		private protected int[]? _maskInt;
		private protected Vector<int>[]? _maskSimd;

		private protected const TypeIndex integerType = TypeIndex.LongIndex;
		private protected const TypeIndex floatType = TypeIndex.IntIndex;

		private int _rho_a;
		private protected int _beta_sbm;
		private protected int _alpha;
		private readonly string networkName = "base";
		private const NetworkType networkType = NetworkType.NotSet;

		private bool _disposed;

//----------------------------------------------------------------------------------------------------------------------

		private protected static bool TopoARTMatchFunction(FTA_F2_node node, long rho)
		{
			return node.MatchValue >= rho;
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt. The input values are internally scaled from [0, 255] to
		/// [0, 1].</param>
		public abstract void Learn(byte[] input);

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public abstract void Learn(decimal[] input);

//----------------------------------------------------------------------------------------------------------------------

		/// <value>Property <c>Alpha</c> represents the choice parameter alpha.</value>
		public decimal Alpha
		{
			get => (decimal)_alpha / Common.ScalingFactor;
			set {
				if(LearningSteps == 0) {
					if(value <= 0.0m) {
						_alpha = ConvertDecimalToInt(0.001m);
						Common.Warning("Too small value for alpha, changed to " + Alpha);
					} else {
						_alpha = ConvertDecimalToInt(value);
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
			get => _beta_sbm / (decimal)Common.ScalingFactor;
			set {
				if(LearningSteps == 0) {
					if(value < 0.0m) {
						_beta_sbm = 0;
						Common.Warning("Too small value for beta_sbm, changed to " + Beta_sbm);
					} else if (value > 1.0m) {
						_beta_sbm = (int)Common.ScalingFactor;
						Common.Warning("Too large value for beta_sbm, changed to " + Beta_sbm);
					} else
						_beta_sbm = ConvertDecimalToInt(value);
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
		public decimal Rho_a
		{
			get => _rho_a / (decimal)Common.ScalingFactor;
			private set {
				if(LearningSteps == 0) {
					if(value < 0.0m) {
						_rho_a = 0;
						Common.Warning("Too small value for rho_a, changed to " + Rho_a);
					} else if (value > 1.0m) {
						_rho_a = (int)Common.ScalingFactor;
						Common.Warning("Too large value for rho_a, changed to " + Rho_a);
					} else
						_rho_a = ConvertDecimalToInt(value);
					Common.Message($"rho_a set to {Rho_a:0.##########}");
				} else
					Common.Warning("Unable to set rho_a after training started, keep old value " + Rho_a);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>IntegerType</c> returns a string containing the data type used for representing integer
		/// variables (IDs, parameters, counters, etc.) internally.</summary>
		public string IntegerType { get; } = Common.Types[(int)integerType];

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Fast_TopoART_base</c>.</summary>
		public decimal FileFormatVersion { get => Common.TopoART_file_format_version; }

		/// <summary>Property <c>FloatType</c> returns a string containing the data type used for representing floating
		/// point variables (input, weights, etc.) internally.</summary>
		public string FloatType { get; } = Common.Types[(int)floatType];

		/// <summary>Property <c>TopoARTFileFormatVersion</c> returns the version of the file format used by class
		/// <c>Fast_TopoART_base</c>.</summary>
		public decimal TopoARTFileFormatVersion { get => FileFormatVersion; }

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Releases unmanaged resources and performs other cleanup operations before the
		/// <see cref="LibTopoART.Fast_TopoART_base"/> is reclaimed by garbage collection.</summary>
		~Fast_TopoART_base()
		{
			Dispose(false);
		}

		/// <summary>Releases all resources used by the <see cref="LibTopoART.Fast_TopoART_base"/> object.</summary>
		/// <remarks>Call <see cref="Dispose()"/> when you are finished using the
		/// <see cref="LibTopoART.Fast_TopoART_base"/>. The <see cref="Dispose()"/> method leaves the
		/// <see cref="LibTopoART.Fast_TopoART_base"/> in an unusable state. After calling <see cref="Dispose()"/>, you
		/// must release all references to the <see cref="LibTopoART.Fast_TopoART_base"/> so the garbage collector can
		/// reclaim the memory that the <see cref="LibTopoART.Fast_TopoART_base"/> was occupying.</remarks>
		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		/// <summary>Release resources used by the <see cref="LibTopoART.Fast_TopoART_base"/> object.</summary>
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

		private protected void InitModules(long inputLen, CreateModule<Fast_TopoART_module, FTA_F2_node, int, Vector<int>, long>
			moduleCreateFunction, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
		{
			long i;
			int rho;

			_modules = new Fast_TopoART_module[ModuleNum];
			for(i = 0, rho = _rho_a; i < ModuleNum; ++i, rho = (rho + (int)Common.ScalingFactor) >> 1) {
				_modules[i] = moduleCreateFunction(inputLen, rho, F2_node_create_func);
			}
		}

		private protected void InitModules(BinaryReader reader, HeaderInfo headerInfo,
			LoadModule<Fast_TopoART_module, FTA_F2_node, Vector<int>, long> moduleLoadFunction,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func,
			LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			_modules = new Fast_TopoART_module[ModuleNum];
			for(long i = 0; i < ModuleNum; ++i) {
				var compatibilityMode = headerInfo.CompatibilityMode(integerType, floatType);
				_modules[i] = moduleLoadFunction(reader, (headerInfo.FileFormatVersions, compatibilityMode), F2_node_create_func, F2_node_load_func);
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

		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t). The input values are internally scaled from [0, 255] to [0, 1].
		/// </param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		public F2_output[] GetBMOutput(byte[] input)
		{
			return GetBMOutput(input, null);
		}

		/// <summary>This method finds the closest category for a given test input.</summary>
		/// <param name="input">The input vector x(t). The input values are internally scaled from [0, 255] to [0, 1].
		/// </param>
		/// <param name="mask">A mask vector excluding individual dimensions of x(t) from the computation. (Setting an
		/// element of the mask vector to <c>true</c>, excludes the corresponding elements of x(t).)</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART module.</returns>
		public F2_output[] GetBMOutput(byte[] input, bool[]? mask)
		{
			Debug.Assert(_x_F0_len == input.LongLength);
			Debug.Assert(_modules != null);

			var result = new F2_output[ModuleNum];

			if(mask != null)
				Debug.Assert(_x_F0_len == mask.LongLength);

			EncodeCurrentInputSimd(input);
			var maskSimd = ConvertMask(mask);

			lock(_learningLock) {
				CompleteLearningQueue();

				for (long i = 0; i < ModuleNum; ++i)
					result[i] = _modules![i].GetBMOutputWithMask(_x_F1_simd!, maskSimd, _phis![i]);
			}

			return result;
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

			var result = new F2_output[ModuleNum];

			if(mask != null)
				Debug.Assert(_x_F0_len == mask.LongLength);

			EncodeCurrentInputSimd(input);
			var maskSimd = ConvertMask(mask);

			lock(_learningLock) {
				CompleteLearningQueue();

				for(long i = 0; i < ModuleNum; ++i)
					result[i] = _modules![i].GetBMOutputWithMask(_x_F1_simd!, maskSimd, _phis![i]);
			}

			return result;
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected Vector<int>[]? ConvertMask(bool[]? mask)
		{
			if(mask == null)
				return null;

			EnsureMaskBuffers();
			Common.FillMaskVectorArray(mask, _maskInt!, _maskSimd!);

			return _maskSimd;
		}

		private protected void EnsureMaskBuffers()
		{
			if(_maskSimd == null) {
				_maskSimd = new Vector<int>[Common.SimdLength<int>(_x_F0_len)];
				_maskInt = new int[_maskSimd.LongLength * Vector<int>.Count];
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private int ConvertInputElement(decimal value, out InputCheckResult checkResult)
		{
			if(value < 0.0m) {
				value = 0.0m;
				checkResult = InputCheckResult.TooLow;
				//Common.Warning("Too small value for input, changed to " + value);
			} else if(value > 1.0m) {
				value = 1.0m;
				checkResult = InputCheckResult.TooHigh;
				//Common.Warning("Too large value for input, changed to " + value);
			} else
				 checkResult = InputCheckResult.OK;

			return (int)(value * Common.ScalingFactor);
		}

		private protected void EncodeCurrentInputSimd(byte[] input)
		{
			_x_F1_simd = new Vector<int>[Common.SimdLength<int>(_x_F0_len) << 1];

			Debug.Assert((Common.SimdLength<int>(input.LongLength) << 1) >= _x_F1_simd!.LongLength);
			Debug.Assert(input.LongLength >= _x_F0_len);

			var dSimd = Common.SimdLength<int>(_x_F0_len);
			long i, j;
			for(i = 0, j = 0; (i < _x_F0_len) && (_x_F0_len - i >= Vector<byte>.Count); i += Vector<byte>.Count) {
				var completeVec = new Vector<byte>(input, (int)i);
				Vector.Widen(completeVec, out Vector<ushort> lowerHalf, out Vector<ushort> higherHalf);
				var lowerHalfConv = Vector.AsVectorInt16(lowerHalf);
				var higherHalfConv = Vector.AsVectorInt16(higherHalf);
				Vector.Widen(lowerHalfConv, out Vector<int> q1, out Vector<int> q2);
				Vector.Widen(higherHalfConv, out Vector<int> q3, out Vector<int> q4);

				q1 = Vector.Multiply(q1, Common.ScalingVectorByteInt);
				q2 = Vector.Multiply(q2, Common.ScalingVectorByteInt);
				q3 = Vector.Multiply(q3, Common.ScalingVectorByteInt);
				q4 = Vector.Multiply(q4, Common.ScalingVectorByteInt);

				_x_F1_simd![j] = q1;
				_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - q1;
				++j;
				_x_F1_simd![j] = q2;
				_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - q2;
				++j;
				_x_F1_simd![j] = q3;
				_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - q3;
				++j;
				_x_F1_simd![j] = q4;
				_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - q4;
				++j;
			}

			if(i < _x_F0_len) {
				var value = new int[Vector<int>.Count];
				for(; (i < _x_F0_len) && (_x_F0_len - i >= Vector<int>.Count); ++j) {
					for(var k = 0; k < Vector<int>.Count; ++i, ++k)
						value[k] = input[i] * (int)Common.ScalingFactorByte;
					var valueVec = new Vector<int>(value);
					_x_F1_simd![j] = valueVec;
					_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - valueVec;
				}

				if(i < _x_F0_len) {
					var mask = new int[Vector<int>.Count];
					for(long k = 0; i < _x_F0_len; ++i, ++k) {
						value[k] = input[i] * (int)Common.ScalingFactorByte;
						mask[k] = -1;
					}
					var valueVec = new Vector<int>(value);
					var maskVec = new Vector<int>(mask);
					_x_F1_simd![j] = Vector.BitwiseAnd(valueVec, maskVec);
					_x_F1_simd![j + dSimd] = Vector.BitwiseAnd(Common.ScalingVectorInt - valueVec, maskVec);
				}
			}
		}

		private protected void EncodeCurrentInputSimd(decimal[] input)
		{
			_x_F1_simd = new Vector<int>[Common.SimdLength<int>(_x_F0_len) << 1];

			Debug.Assert((Common.SimdLength<int>(input.LongLength) << 1) >= _x_F1_simd!.LongLength);
			Debug.Assert(input.LongLength >= _x_F0_len);

			var dSimd = Common.SimdLength<int>(_x_F0_len);

			var value = new int[Vector<int>.Count];
			var adaptedInput = false;
			long i, j;
			for(i = 0, j = 0; (i < _x_F0_len) && (_x_F0_len - i >= Vector<int>.Count); ++j) {
				for(var k = 0; k < Vector<int>.Count; ++i, ++k) {
					value[k] = ConvertInputElement(input[i], out InputCheckResult checkResult);
					adaptedInput = checkResult == InputCheckResult.OK ? adaptedInput : true;
				}
				var valueVec = new Vector<int>(value);
				_x_F1_simd![j] = valueVec;
				_x_F1_simd![j + dSimd] = Common.ScalingVectorInt - valueVec;
			}

			if(i < _x_F0_len) {
				var mask = new int[Vector<int>.Count];
				for(long k = 0; i < _x_F0_len; ++i, ++k) {
					value[k] = ConvertInputElement(input[i], out InputCheckResult checkResult);
					adaptedInput = checkResult == InputCheckResult.OK ? adaptedInput : true;
					mask[k] = -1;
				}
				var valueVec = new Vector<int>(value);
				var maskVec = new Vector<int>(mask);
				_x_F1_simd![j] = Vector.BitwiseAnd(valueVec, maskVec);
				_x_F1_simd![j + dSimd] = Vector.BitwiseAnd(Common.ScalingVectorInt - valueVec, maskVec);
			}

			if(adaptedInput)
				Common.Warning(Common.Warning_CorrectedInputValue, VerbosityLevel.Verbose);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected void SetTopoARTParams(long inputLen, long moduleNum, decimal rho_a)
		{
			if(!Vector.IsHardwareAccelerated)
				Common.Warning("Hardware acceleration not supported (computation speed may be significantly reduced)", VerbosityLevel.Important);

			InitialisationMessage();

			if(inputLen < 1) {
				inputLen = 1;				// check required, as otherwise HTA causes a division by zero
				Common.Warning("Invalid length of the input vector x_F0, changed to " + inputLen);
			}
			_x_F0_len = inputLen;

			// set default x_F1_simd
			_x_F1_simd = new Vector<int>[Common.SimdLength<int>(inputLen) << 1];

			if(rho_a < 0.0m) {
				_rho_a = 0;
				Common.Warning("Too small value for rho_a, changed to " + Rho_a);
			} else if (rho_a > 1.0m) {
				_rho_a = (int)Common.ScalingFactor;
				Common.Warning("Too large value for rho_a, changed to " + Rho_a);
			} else
				_rho_a = ConvertDecimalToInt(rho_a);

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

//----------------------------------------------------------------------------------------------------------------------

		private protected BinaryReader LoadTopoARTParams(Stream stream, MatchFunction<FTA_F2_node, long>? initialMatchFunction,
			out HeaderInfo headerInfo)
		{
			long tmpLearningSteps;

			using(var headerReader = new BinaryReader(stream, Encoding.UTF8, true))
				headerInfo = LoadBinaryHeader(headerReader);

			var reader = headerInfo.Flags == SaveFlags.None ? new BinaryReader(stream, Encoding.UTF8, true) : new BinaryReader(new GZipStream(stream, CompressionMode.Decompress, true));

			try {
				var compatibilityMode = headerInfo.CompatibilityMode(integerType, floatType);
				LoadPrecedingBinaryInformation(reader, (headerInfo.FileFormatVersions, compatibilityMode));

				_x_F0_len = reader.ReadInt64();

				_x_F1_simd = new Vector<int>[Common.SimdLength<int>(_x_F0_len) << 1];

				long tmpPhi;
				if(compatibilityMode) {
					for(long i = 0; i < _x_F0_len; ++i)
						ConvertDecimalToInt(reader.ReadDecimal());

					Rho_a = reader.ReadDecimal();
					Beta_sbm = reader.ReadDecimal();
					Tau = reader.ReadInt64();
					tmpPhi = reader.ReadInt64();
					tmpLearningSteps = reader.ReadInt64();
					Alpha = reader.ReadDecimal();
				} else {
					for(long i = 0; i < _x_F0_len; ++i)
						reader.ReadInt32();

					_rho_a = reader.ReadInt32();
					_beta_sbm = reader.ReadInt32();
					Tau = reader.ReadInt64();
					tmpPhi = reader.ReadInt64();
					tmpLearningSteps = reader.ReadInt64();
					_alpha = reader.ReadInt32();
				}

				_skipEdgeLearning = headerInfo.FileFormatVersions.TopoARTFileFormatVersion >= 1.0m ? reader.ReadBoolean() : false;

				ModuleNum	=	reader.ReadInt64();
				_phis		=	new long[ModuleNum];

				// load phi array if required
				if(tmpPhi == LibTopoART_info.UNDEFINED) {
					for(long i = 0; i < ModuleNum; ++i)
						_phis[i] = reader.ReadInt64();
				}
				else
					Phi = tmpPhi;

				LearningSteps = tmpLearningSteps;

				return reader;
			} catch {
				reader.Dispose();
				throw;
			}
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
			Debug.Assert(_x_F1_simd != null);
			Debug.Assert(_modules != null);

			SaveTextHeader(writer);
			SavePrecedingTextInformation(writer);

			writer.WriteLine("x^F0 length: " + _x_F0_len);
			writer.Write("x^F0:");
			for(long i = 0; i < _x_F0_len; ++i) {
				var (i1, i2) = Common.SimdIndexes<int>(i, _x_F0_len);
				writer.Write(" " + (_x_F1_simd![i1][i2] / (decimal)Common.ScalingFactor).ToString(CultureInfo.InvariantCulture));
			}
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

		/// <summary>This method saves the entire network as a binary file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		/// <param name="compression">Compression level of the save file (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(string path, CompressionLevel compression = CompressionLevel.Fastest)
		{
#if DEBUG
			if(TopoARTFileFormatVersion == 0.09m)
				Save(path, true, CompressionLevel.NoCompression);
			else
#endif
				if(TopoARTFileFormatVersion == 0.10m)
					Save(path, false, CompressionLevel.NoCompression);
				else if(TopoARTFileFormatVersion >= 0.11m)
					Save(path, false, compression);
		}

		/// <summary>This method saves the entire network to a stream using the binary file format. The stream is left
		/// open.</summary>
		/// <param name="stream">A writable <c>Stream</c> the network is saved to.</param>
		/// <param name="compression">Compression level of the saved data (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(Stream stream, CompressionLevel compression = CompressionLevel.Fastest)
		{
#if DEBUG
			if(TopoARTFileFormatVersion == 0.09m)
				Save(stream, true, CompressionLevel.NoCompression);
			else
#endif
				if(TopoARTFileFormatVersion == 0.10m)
					Save(stream, false, CompressionLevel.NoCompression);
				else if(TopoARTFileFormatVersion >= 0.11m)
					Save(stream, false, compression);
		}

		/// <summary>This method saves the entire network as a binary file.</summary>
		/// <param name="path">A <c>string</c> representing the path of the file to save.</param>
		/// <param name="compatibilityMode">If true, the file is saved in compatibility mode.</param>
		/// <param name="compression">Compression level of the save file (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(string path, bool compatibilityMode, CompressionLevel compression = CompressionLevel.Fastest)
		{
			using var file = File.Open(path, FileMode.Create);
			Save(file, compatibilityMode, compression);
		}

		/// <summary>This method saves the entire network to a stream using the binary file format. The stream is left
		/// open.</summary>
		/// <param name="stream">A writable <c>Stream</c> the network is saved to.</param>
		/// <param name="compatibilityMode">If true, the network is saved in compatibility mode.</param>
		/// <param name="compression">Compression level of the saved data (Compression is not supported by LibTopoART
		/// v0.93 and below.)</param>
		public void Save(Stream stream, bool compatibilityMode, CompressionLevel compression = CompressionLevel.Fastest)
		{
			Debug.Assert(_x_F1_simd != null);
			Debug.Assert(_modules != null);

			using(var headerWriter = new BinaryWriter(stream, Encoding.UTF8, true))
				SaveBinaryHeader(headerWriter, compatibilityMode, compression);

			using var writer = compression == CompressionLevel.NoCompression ? new BinaryWriter(stream, Encoding.UTF8, true) : new BinaryWriter(new GZipStream(stream, compression, true));

			SavePrecedingBinaryInformation(writer, compatibilityMode);

			writer.Write(_x_F0_len);

			if(compatibilityMode) {
				for(long i = 0; i < _x_F0_len; ++i) {
					var (i1, i2) = Common.SimdIndexes<int>(i, _x_F0_len);
					writer.Write((_x_F1_simd![i1][i2] / (decimal)Common.ScalingFactor));
				}
				writer.Write(Rho_a);
				writer.Write(Beta_sbm);
				writer.Write(Tau);
				writer.Write(Phi);
				writer.Write(LearningSteps);
				writer.Write(Alpha);
			} else {
				for(long i = 0; i < _x_F0_len; ++i) {
					var (i1, i2) = Common.SimdIndexes<int>(i, _x_F0_len);
					writer.Write(_x_F1_simd![i1][i2]);
				}
				writer.Write(_rho_a);
				writer.Write(_beta_sbm);
				writer.Write(Tau);
				writer.Write(Phi);
				writer.Write(LearningSteps);
				writer.Write(_alpha);
			}

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
					_modules![i].Save(writer, compatibilityMode);
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected virtual HeaderInfo LoadBinaryHeader(BinaryReader reader)
		{
			return Common.LoadBinaryHeader(reader, networkType, networkName, FileFormatVersion, integerType, floatType);
		}

		private protected virtual void LoadPrecedingBinaryInformation(BinaryReader reader, (FileFormatVersions, bool) fileFormatInfo) {}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*        TopoART network        *");
			writer.WriteLine("*      LibTopoART (v" + LibTopoART_info.version.ToString(CultureInfo.InvariantCulture) + ")       *");
			writer.WriteLine("*********************************");
			writer.WriteLine("file format version: " + FileFormatVersion.ToString(CultureInfo.InvariantCulture));
			writer.WriteLine("integer type: " + IntegerType);
			writer.WriteLine("float type: " + FloatType);
			writer.WriteLine("*********************************");
		}

		private protected override void SavePrecedingTextInformation(TextWriter writer) {}

		private protected virtual void SaveBinaryHeader(BinaryWriter writer, bool compatibilityMode, CompressionLevel compression)
		{
			if(compatibilityMode)
				Common.SaveCompatibleBinaryHeader(writer, networkType, FileFormatVersion, compression);
			else
				Common.SaveBinaryHeader(writer, networkType, FileFormatVersion, (integerType, floatType), compression);
		}

		private protected virtual void SavePrecedingBinaryInformation(BinaryWriter writer, bool compatibilityMode) {}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method resets the adaptation state to <c>AdaptationState.NO_ADAPTATION</c>.</summary>
		/// <exception cref="InvalidNumberException">Thrown when the number of edges of an F2 node is greater than
		/// <c>int.MaxValue</c>.</exception>
		public void ResetAdaptationState()
		{
			ResetAdaptationState<Fast_TopoART_module, Vector<int>>(_modules);
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

				int eps = (int)(epsilon * Common.ScalingFactor);
				AdaptationState result = AdaptationState.NO_ADAPTATION;

				for(long i = 0; i < ModuleNum; ++i) {
					result |= _modules![i].GetAdaptationState(_phis![i], (weight1, weight2) =>  {
						Vector<int> diff = Vector.Abs(weight2 - weight1);
						return Vector.GreaterThanAny(diff, eps * Vector<int>.One);
					});
				}

				return result;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected static int ConvertDecimalToInt(decimal value)
		{
			return (int)Math.Max(Math.Min(value * Common.ScalingFactor, int.MaxValue), int.MinValue);
		}

		private protected static byte ConvertLongToByte(long value)
		{
			return (byte)Math.Max(Math.Min((value / Common.ScalingFactorByte), byte.MaxValue), byte.MinValue);
		}

		private protected static decimal ConvertLongToDecimal(long value)
		{
			return value / (decimal)Common.ScalingFactor;
		}

		private protected virtual void InitialisationMessage()
		{
			Common.InitialisationMessage(networkName);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method checks a provided module index and sets <c>FINAL_MODULE</c> to <c>ModuleNum - 1</c>.
		/// </summary>
		/// <param name="moduleIndex">The module index to be checked.</param>
		/// <exception cref="InvalidModuleIndexException">Thrown when <paramref name="moduleIndex"/> is invalid.
		/// </exception>
		private protected long GetModuleIndex(long moduleIndex)
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