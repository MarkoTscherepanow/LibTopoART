/*"*********************************************************************************************************************
*                                                   TopoART-AM class                                                   *
*                                     created by Marko Tscherepanow, 17 March 2018                                     *
************************************************************************************************************************
*                               $Id: Fast_TopoART_AM.cs 1832 2026-07-17 06:24:27Z marko $                              *
***********************************************************************************************************************/

using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Fast_TopoART_AM</c> provides an implementation of the TopoART-AM neural network as proposed in
	/// "Marko Tscherepanow, Marco Kortkamp and Marc Kammer (2011). A Hierarchical ART Network for the Stable
	/// Incremental Learning of Topological Structures and Associations from Noisy Data. Neural Networks 24(8): 906-916.
	/// Elsevier."
	/// <para>Class <c>Fast_TopoART_AM</c> requires all input and output to lie in the interval [0, 1].</para>
	/// </summary>
	public class Fast_TopoART_AM : Fast_TopoART, IFast_TopoART_AM
	{
		private enum RecallKey { None, Key1, Key2 }

		private Vector<int>[]? _recallMask1Simd;
		private Vector<int>[]? _recallMask2Simd;
		private RecallKey _recallKey;
		private long _recallModuleIndex;
		private const string _networkName = "TopoART-AM";
		private const NetworkType _networkType = NetworkType.TopoARTAM;

		private byte[]? _kVecByte;
		private decimal[]? _kVecDecimal;

		private byte[]? _x_F0_byte;
		private decimal[]? _x_F0_decimal;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Fast_TopoART_AM</c>.</summary>
		public new decimal FileFormatVersion { get => Common.TopoART_AM_file_format_version; }

		/// <summary>Property <c>Key1Len</c> returns the length of the first key vector.</summary>
		public long Key1Len { get; private set; }

		/// <summary>Property <c>Key2Len</c> returns the length of the second key vector.</summary>
		public long Key2Len { get; private set; }

//----------------------------------------------------------------------------------------------------------------------

		private Fast_TopoART_module CreateTopoARTAMModule(long inputLen, int rho, CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func)
		{
			return new Fast_TopoART_AM_module(inputLen, rho, F2_node_create_func);
		}

		private Fast_TopoART_module LoadTopoARTAMModule(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo,
			CreateF2Node<FTA_F2_node, Vector<int>, long>? F2_node_create_func, LoadF2Node<FTA_F2_node> F2_node_load_func)
		{
			return new Fast_TopoART_AM_module(reader, fileFormatInfo, F2_node_create_func, F2_node_load_func);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This constructor initialises a TopoART-AM network.</summary>
		/// <param name="key1Len">The length of the first key vector to be learnt.</param>
		/// <param name="key2Len">The length of the second key vector to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART-AM modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART-AM module (TopoART-AM a).</param>
		public Fast_TopoART_AM(long key1Len, long key2Len, long moduleNum, decimal rho_a)
		{
			Key1Len = CheckLength(key1Len);
			Key2Len = CheckLength(key2Len);

			SetTopoARTParams(Key1Len + Key2Len, moduleNum, rho_a);

			if(Key1Len != key1Len)
				Common.Warning("Invalid length of key 1, changed to " + Key1Len);

			if(Key2Len != key2Len)
				Common.Warning("Invalid length of key 2, changed to " + Key2Len);

			InitModules((Key1Len + Key2Len) << 1, CreateTopoARTAMModule, CreateTopoARTF2Node);

			InitTransientMembers();
		}

		/// <summary>This constructor loads a saved TopoART-AM network.</summary>
		/// <param name="path">The path of a binary TopoART-AM file.</param>
		/// <exception cref="InvalidFileException">Thrown when the given file cannot be loaded.</exception>
		public Fast_TopoART_AM(string path)
		{
			using(var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read))
				LoadNetwork(file);

			InitTransientMembers();
		}

		/// <summary>This constructor loads a saved TopoART-AM network from a stream. The stream is left open.
		/// </summary>
		/// <param name="stream">A readable <c>Stream</c> containing a network in the binary TopoART-AM file format.
		/// </param>
		/// <exception cref="InvalidFileException">Thrown when the given stream cannot be loaded.</exception>
		public Fast_TopoART_AM(Stream stream)
		{
			LoadNetwork(stream);
			InitTransientMembers();
		}

		private void LoadNetwork(Stream stream)
		{
			using var reader = LoadTopoARTParams(stream, TopoARTMatchFunction, out var headerInfo);
			InitModules(reader, headerInfo, LoadTopoARTAMModule, CreateTopoARTF2Node, LoadTopoARTF2Node);
		}

		private void InitTransientMembers()
		{
			var recallMask1 = new int[Key1Len + Key2Len];
			var recallMask2 = new int[Key1Len + Key2Len];

			// create excitatory mask vector (-1 == 0xffffffff)
			for(long i = 0; i < Key1Len; ++i) {
				recallMask1[i] = 0;
				recallMask2[i] = -1;
			}
			for(long i = 0; i < Key2Len; ++i) {
				recallMask1[Key1Len + i] = -1;
				recallMask2[Key1Len + i] = 0;
			}

			_recallMask1Simd = Common.CreateVectorArray(recallMask1);
			_recallMask2Simd = Common.CreateVectorArray(recallMask2);

			_recallKey = RecallKey.None;
			_recallModuleIndex = FINAL_MODULE;

			_kVecByte = new byte[Key1Len + Key2Len];
			_kVecDecimal = new decimal[Key1Len + Key2Len];

			_x_F0_byte = new byte[Key1Len + Key2Len];
			_x_F0_decimal = new decimal[Key1Len + Key2Len];
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method finds the closest category for a given pair of keys.</summary>
		/// <param name="key1">The first key vector. The elements of the key vector are internally scaled from [0, 255]
		/// to [0, 1].</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>. The elements of the key
		/// vector are internally scaled from [0, 255] to [0, 1].</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART-AM module.</returns>
		public F2_output[] GetBMOutput(byte[] key1, byte[] key2)
		{
			Debug.Assert(_kVecByte != null);
			key1.CopyTo(_kVecByte, 0);
			key2.CopyTo(_kVecByte, Key1Len);
			return GetBMOutput(_kVecByte!);
		}

		/// <summary>This method finds the closest category for a given pair of keys.</summary>
		/// <param name="key1">The first key vector.</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>.</param>
		/// <returns>An array of type <c>F2_output</c>. Each entry contains the ID of the best-matching node and the
		/// corresponding cluster ID for one TopoART-AM module.</returns>
		public F2_output[] GetBMOutput(decimal[] key1, decimal[] key2)
		{
			Debug.Assert(_kVecDecimal != null);
			key1.CopyTo(_kVecDecimal, 0);
			key2.CopyTo(_kVecDecimal, Key1Len);
			return GetBMOutput(_kVecDecimal!);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step.</summary>
		/// <param name="key1">The first key vector to be learnt.</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>.</param>
		public void Learn(byte[] key1, byte[] key2)
		{
			Debug.Assert(key1.LongLength == Key1Len);
			Debug.Assert(key2.LongLength == Key2Len);
			Debug.Assert(_kVecByte != null);

			key1.CopyTo(_kVecByte, 0);
			key2.CopyTo(_kVecByte, Key1Len);
			LearnWithMask(_kVecByte!, null);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="key1">The first key vector to be learnt. The elements of the key vector are internally scaled
		/// from [0, 255] to [0, 1].</param>
		/// <param name="key2">The second key vector corresponding to <paramref name="key1"/>. The elements of the key
		/// vector are internally scaled from [0, 255] to [0, 1].</param>
		public void Learn(decimal[] key1, decimal[] key2)
		{
			Debug.Assert(key1.LongLength == Key1Len);
			Debug.Assert(key2.LongLength == Key2Len);
			Debug.Assert(_kVecDecimal != null);

			key1.CopyTo(_kVecDecimal, 0);
			key2.CopyTo(_kVecDecimal, Key1Len);
			LearnWithMask(_kVecDecimal!, null);
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
			Key1Len = reader.ReadInt64();
			Key2Len = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*      TopoART-AM network       *");
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
			writer.WriteLine("key 1 length: " + Key1Len);
			writer.WriteLine("key 2 length: " + Key2Len);
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
			writer.Write(Key1Len);
			writer.Write(Key2Len);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method starts the recall process for the first key vector.</summary>
		/// <param name="key2">The stimulus (second key vector) which is used to trigger recall. The stimulus elements
		/// are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall. (<c>FINAL_MODULE</c> denotes
		/// the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		/// <exception cref="InvalidModuleIndexException">Thrown when
		/// <paramref name="moduleIndex"/> is invalid.</exception>
		public long BeginRecallKey1(byte[] key2, long moduleIndex = FINAL_MODULE)
		{
			Debug.Assert(key2.LongLength == Key2Len);
			Debug.Assert(_x_F0_byte != null);
			Debug.Assert(_recallMask1Simd != null);
			Debug.Assert(_modules != null);

			_recallKey = RecallKey.Key1;
			_recallModuleIndex = GetModuleIndex(moduleIndex);

			key2.CopyTo(_x_F0_byte, Key1Len);

			EncodeCurrentInputSimd(_x_F0_byte!);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).BeginRecall(_x_F1_simd!, _recallMask1Simd!, _phis![_recallModuleIndex]);
			}
		}

		/// <summary>This method starts the recall process for the first key vector.</summary>
		/// <param name="key2">The stimulus (second key vector) which is used to trigger recall.</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall. (<c>FINAL_MODULE</c> denotes
		/// the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		/// <exception cref="InvalidModuleIndexException">Thrown when
		/// <paramref name="moduleIndex"/> is invalid.</exception>
		public long BeginRecallKey1(decimal[] key2, long moduleIndex = FINAL_MODULE)
		{
			Debug.Assert(key2.LongLength == Key2Len);
			Debug.Assert(_x_F0_decimal != null);
			Debug.Assert(_recallMask1Simd != null);
			Debug.Assert(_modules != null);

			_recallKey = RecallKey.Key1;
			_recallModuleIndex = GetModuleIndex(moduleIndex);

			key2.CopyTo(_x_F0_decimal, Key1Len);
			EncodeCurrentInputSimd(_x_F0_decimal!);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).BeginRecall(_x_F1_simd!, _recallMask1Simd!, _phis![_recallModuleIndex]);
			}
		}

		/// <summary>This method starts the recall process for the second key vector.</summary>
		/// <param name="key1">The stimulus (first key vector) which is used to trigger recall. The stimulus elements
		/// are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall. (<c>FINAL_MODULE</c> denotes
		/// the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		/// <exception cref="InvalidModuleIndexException">Thrown when
		/// <paramref name="moduleIndex"/> is invalid.</exception>
		public long BeginRecallKey2(byte[] key1, long moduleIndex = FINAL_MODULE)
		{
			Debug.Assert(key1.LongLength == Key1Len);
			Debug.Assert(_x_F0_byte != null);
			Debug.Assert(_recallMask2Simd != null);
			Debug.Assert(_modules != null);
			Debug.Assert(_x_F1_simd != null);

			_recallKey = RecallKey.Key2;
			_recallModuleIndex = GetModuleIndex(moduleIndex);

			key1.CopyTo(_x_F0_byte, 0);
			EncodeCurrentInputSimd(_x_F0_byte!);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).BeginRecall(_x_F1_simd!, _recallMask2Simd!, _phis![_recallModuleIndex]);
			}
		}

		/// <summary>This method starts the recall process for the second key vector.</summary>
		/// <param name="key1">The stimulus (first key vector) which is used to trigger recall.</param>
		/// <param name="moduleIndex">Index of the TopoART-AM module to be used for recall. (<c>FINAL_MODULE</c> denotes
		/// the module with the highest index.)</param>
		/// <returns>The number of F3 nodes created.</returns>
		/// <exception cref="InvalidModuleIndexException">Thrown when
		/// <paramref name="moduleIndex"/> is invalid.</exception>
		public long BeginRecallKey2(decimal[] key1, long moduleIndex = FINAL_MODULE)
		{
			Debug.Assert(key1.LongLength == Key1Len);
			Debug.Assert(_x_F0_decimal != null);
			Debug.Assert(_recallMask2Simd != null);
			Debug.Assert(_modules != null);
			Debug.Assert(_x_F1_simd != null);

			_recallKey = RecallKey.Key2;
			_recallModuleIndex = GetModuleIndex(moduleIndex);

			key1.CopyTo(_x_F0_decimal, 0);
			EncodeCurrentInputSimd(_x_F0_decimal!);

			lock(_learningLock) {
				CompleteLearningQueue();
				return ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).BeginRecall(_x_F1_simd!, _recallMask2Simd!, _phis![_recallModuleIndex]);
			}
		}

		/// <summary>This method performs a single associative recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step. The elements of the recall result
		/// are internally scaled from [0, 1] to [0, 255].</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed or not.</returns>
		public bool RecallStep(out byte[]? recallResult, out decimal F3_activation)
		{
			Debug.Assert(_modules != null);

			if(_recallModuleIndex >= 0) {
				var success = ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).RecallStep(out byte[]? cog, out decimal act);
				return RecallStepCommon(success, cog, act, out recallResult, out F3_activation);
			}

			return RecallStepCommon(false, null, LibTopoART_info.UNDEFINED, out recallResult, out F3_activation);
		}

		/// <summary>This method performs a single associative recall step.</summary>
		/// <param name="recallResult">Returns the recall output for the current step.</param>
		/// <param name="F3_activation">Returns the activation of the current F3 node.</param>
		/// <returns>A boolean result indicating whether the recall step was successfully completed or not.</returns>
		public bool RecallStep(out decimal[]? recallResult, out decimal F3_activation)
		{
			Debug.Assert(_modules != null);

			if(_recallModuleIndex >= 0) {
				var success = ((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).RecallStep(out decimal[]? cog, out decimal act);
				return RecallStepCommon(success, cog, act, out recallResult, out F3_activation);
			}

			return RecallStepCommon(false, null, LibTopoART_info.UNDEFINED, out recallResult, out F3_activation);
		}

		private bool RecallStepCommon<TType>(bool success, TType[]? cog, decimal act, out TType[]? recallResult,
			out decimal F3_activation) where TType : struct
		{
			recallResult = null;
			F3_activation = LibTopoART_info.UNDEFINED;

			if(cog == null)
				success = false;

			if(success) {
				if(_recallKey == RecallKey.Key1) {
					recallResult = new TType[Key1Len];
					Array.Copy(cog!, 0, recallResult, 0, Key1Len);
				}
				else if(_recallKey == RecallKey.Key2) {
					recallResult = new TType[Key2Len];
					Array.Copy(cog!, Key1Len, recallResult, 0, Key2Len);
				}
				else
					success = false;
			}

			if(success)
				F3_activation = act;

			Debug.Assert((success && recallResult != null) || (!success && recallResult == null && F3_activation == LibTopoART_info.UNDEFINED));

			return success;
		}

		/// <summary>This method stops the recall process and frees temporary resources.</summary>
		public void EndRecall()
		{
			Debug.Assert(_recallModuleIndex >= 0);
			Debug.Assert(_modules != null);

		 	if((_recallModuleIndex >= 0))
				((Fast_TopoART_AM_module)_modules![_recallModuleIndex]).EndRecall();

			_recallKey = RecallKey.None;
			_recallModuleIndex = FINAL_MODULE;
		}
	}

//**********************************************************************************************************************

}