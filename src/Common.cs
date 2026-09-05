using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Threading.Tasks;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Text;

// Public Key Token: 21f0a892fb9d8265
[assembly: InternalsVisibleTo("LibTopoART.Compatibility, PublicKey=" +
	"00240000048000009400000006020000002400005253413100040000010001005d7e23e3ccb668" +
	"1ff96065876df14e4d0b567976dd410a56294fab7e8767084b13f4862c53ae12f5dc2682fee73a" +
	"d2de564fd1b8b6d5ce4f11b445c0fcd6dd2305cc76cf05db97bd741237e133b91c339b44edf036" +
	"6a23c0f7a8f3da64ff66bb0e2976cf7aac557cf64c7739fa303ab654d8aa66c1dd53670808f972" +
	"1b12a685")]

namespace LibTopoART
{
#pragma warning disable 1591

//**********************************************************************************************************************

	internal enum ActivationType
	{
		Training,
		Prediction
	}

	internal enum LearningResult
	{
		DoNotPropagate,
		PropagateFurther
	}

#if DEBUG
	/// <summary>Enumeration specifying save flags.</summary>
	[Flags]
	public enum SaveFlags : uint
#else
	[Flags]
	internal enum SaveFlags : uint
#endif
	{
		None = 0,
		GZip = 1,
	}

#if DEBUG
	/// <summary>Enumeration specifying types of possible neural networks.</summary>
	public enum NetworkType : uint
#else
	internal enum NetworkType : uint
#endif
	{
		NotSet					=	0x000000,
		TopoART					=	0x000001,
		HypersphereTopoART		=	0x000002,
		Episodic				=	0x000100,
		AssociativeMemory		=	0x001000,
		Classification			=	0x010000,
		Regression				=	0x100000,

		EpisodicTopoART			=	Episodic + TopoART,

		HypersphereTopoARTC		=	Classification + HypersphereTopoART,

		TopoARTAM				=	AssociativeMemory + TopoART,
		TopoARTC				=	Classification + TopoART,
		TopoARTR				=	Regression + TopoART
	}

#if DEBUG
	/// <summary>Enumeration specifying possible integer and floating point types. The values are
	/// indexes to <c>Common.types.</c></summary>
	public enum TypeIndex : uint
#else
	internal enum TypeIndex : uint
#endif
	{
		SbyteIndex		=	0,
		ByteIndex		=	1,
		ShortIndex		=	2,
		UshortIndex		=	3,
		IntIndex		=	4,
		UintIndex		=	5,
		LongIndex		=	6,
		UlongIndex		=	7,
		FloatIndex		=	8,
		DoubleIndex		=	9,
		DecimalIndex	=	10
	}

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Struct containing and handling file format versions.</summary>
	public readonly struct FileFormatVersions
#else
	internal readonly struct FileFormatVersions
#endif
	{
		public ushort HeaderOffset { get; }
		public decimal FileFormatVersion { get; }
		public decimal BaseFileFormatVersion { get; }
		public decimal TopoARTFileFormatVersion { get; }
		public bool IsTopoART { get; }

		public FileFormatVersions(in decimal taFileFormatVersion)
		{
			FileFormatVersion			=	taFileFormatVersion;
			BaseFileFormatVersion		=	taFileFormatVersion;
			TopoARTFileFormatVersion	=	taFileFormatVersion;
			IsTopoART = true;
			HeaderOffset = 1;
		}

		public FileFormatVersions(in decimal fileFormatVersion, in decimal taFileFormatVersion)
		{
			FileFormatVersion			=	fileFormatVersion;
			BaseFileFormatVersion		=	taFileFormatVersion;
			TopoARTFileFormatVersion	=	taFileFormatVersion;
			IsTopoART = false;
			HeaderOffset = 2;
		}

		public FileFormatVersions(in decimal fileFormatVersion, in decimal baseFileFormatVersion, in decimal taFileFormatVersion)
		{
			FileFormatVersion			=	fileFormatVersion;
			BaseFileFormatVersion		=	baseFileFormatVersion;
			TopoARTFileFormatVersion	=	taFileFormatVersion;
			IsTopoART = false;
			HeaderOffset = 3;
		}
	}

#if DEBUG
	/// <summary>Struct containing and handling file header data.</summary>
	public readonly struct HeaderInfo(decimal version, NetworkType type, in FileFormatVersions fileFormatVersions, SaveFlags flags, TypeIndex? intType, TypeIndex? floatType)
#else
	internal readonly struct HeaderInfo(decimal version, NetworkType type, in FileFormatVersions fileFormatVersions, SaveFlags flags, TypeIndex? intType, TypeIndex? floatType)
#endif
	{
		public FileFormatVersions FileFormatVersions { get; } = fileFormatVersions;
		public SaveFlags Flags { get; } = flags;
		public TypeIndex? FloatType { get; } = floatType;
		public TypeIndex? IntType { get; } = intType;
		public NetworkType Type { get; } = type;
		public decimal Version { get; } = version;

		public bool CompatibilityMode(TypeIndex refIntType, TypeIndex refFloatType)
		{
			if(FileFormatVersions.TopoARTFileFormatVersion >= 0.10m) {
				var compatibilityModeI = false;
				var compatibilityModeF = false;

				if(IntType != refIntType) {
					if(IntType != TypeIndex.LongIndex)
						throw new InvalidFileException(Common.InvalidFileException_UnsupportedIntegerType);
					else
						compatibilityModeI = true;
				}

				if(FloatType != refFloatType) {
					if(FloatType != TypeIndex.DecimalIndex)
						throw new InvalidFileException(Common.InvalidFileException_UnsupportedFloatType);
					else
						compatibilityModeF = true;
				}

				return (compatibilityModeI || compatibilityModeF);
			} else
				return true;
		}
	}

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Delegate representing functions comparing two individual weights.</summary>
	public delegate bool CompareWeights<TFloatType>(TFloatType weight1, TFloatType weight2);
#else
	internal delegate bool CompareWeights<TFloatType>(TFloatType weight1, TFloatType weight2);
#endif

	internal delegate TModuleType CreateModule<TModuleType, TNodeType,  TFloatType, TSpatialWeightType,
		TTemporalWeightType>(long inputLen, TFloatType rho,
		 CreateF2Node<TNodeType, TSpatialWeightType, TTemporalWeightType>? F2_node_create_func);
	internal delegate TModuleType LoadModule<TModuleType, TNodeType, TSpatialWeightType, TTemporalWeightType>
		(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo,
		 CreateF2Node<TNodeType, TSpatialWeightType, TTemporalWeightType>? F2_node_create_func,
		 LoadF2Node<TNodeType> F2_node_load_func);
	internal delegate TNodeType CreateF2Node<TNodeType, TSpatialWeightType, TTemporalWeightType>
		(long nodeID, long inputLen, TSpatialWeightType[] spatialWeights,
		 TTemporalWeightType[]? temporalWeights);

	internal delegate TNodeType LoadF2Node<TNodeType>(BinaryReader reader, in (FileFormatVersions, bool) fileFormatInfo);

	internal delegate bool MatchFunction<in TNodeType, TFloatType>(TNodeType node, TFloatType rho);

	internal delegate void ModuleFunction(long moduleIndex);

//**********************************************************************************************************************

	internal class ActivationThreadProvider<TFloatType, TSpatialWeightType, TTemporalWeightType,
		TMaskType> : IDisposable where TFloatType : new()
	{
		private readonly IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>?[] _threadArray =
			new IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>[LibTopoART_info.MaximumThreads];
		private readonly Task[] _tasks = new Task[LibTopoART_info.MaximumThreads];

		private long _threadNodeNum;
		private long _serialWorkLimit;
		private long _serialPerNodeWorkOffset;

		private bool _isActive;
		private bool _disposed;

//----------------------------------------------------------------------------------------------------------------------

		~ActivationThreadProvider()
		{
			Dispose(false);
		}

		public void Dispose()
		{
			Dispose(true);
			GC.SuppressFinalize(this);
		}

		protected virtual void Dispose(bool disposing)
		{
			if(!_disposed)
			{
				if(disposing)
				{
					Debug.Assert(_tasks != null);
					if(_tasks != null) {
						foreach(var task in _tasks)
							task?.Dispose();
					}
				}

				_disposed = true;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		protected void InitThreads(long serialWorkLimit = 0, long serialPerNodeWorkOffset = 0)
		{
			_serialWorkLimit = serialWorkLimit;
			_serialPerNodeWorkOffset = serialPerNodeWorkOffset;
			_isActive = true;
		}

		protected void InsertIntoThreadArray(IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType,
			TMaskType> newNode, long threadID)
		{
			if(_isActive) {
				if(_threadArray[threadID] != null)
					_threadArray[threadID]!.ThreadPrev = newNode;
				newNode.ThreadNext = _threadArray[threadID];
				newNode.ThreadID = threadID;
				_threadArray[threadID] = newNode;
				++_threadNodeNum;
			}
		}

		protected void RemoveFromThreadArray(IF2_node_threading<TFloatType, TSpatialWeightType, TTemporalWeightType,
			TMaskType> node)
		{
			if(_isActive) {
				if(node.ThreadPrev != null)
					node.ThreadPrev.ThreadNext = node.ThreadNext;
				else
					_threadArray[node.ThreadID] = node.ThreadNext;

				if(node.ThreadNext != null)
					node.ThreadNext.ThreadPrev = node.ThreadPrev;

				--_threadNodeNum;
			}
		}

		protected void RunActivationThreads(TFloatType alpha, TMaskType[]? mask, TSpatialWeightType[] x_F1_vec,
											ActivationType actType = ActivationType.Training)
		{
			RunActivationThreads(alpha, mask, x_F1_vec, null, actType);
		}

		protected void RunActivationThreads(TFloatType alpha, TMaskType[]? mask, TSpatialWeightType[] x_F1_vec,
											TTemporalWeightType[]? t_F1_vec, ActivationType actType = ActivationType.Training)
		{
			if(_isActive) {
				// serial path for small workloads
				if(_threadNodeNum * (x_F1_vec.LongLength + _serialPerNodeWorkOffset) <= _serialWorkLimit) {
					switch(actType) {
						case ActivationType.Training:
								for(long currentTask = 0; currentTask < LibTopoART_info.MaximumThreads; ++currentTask) {
									for(var currentNode = _threadArray[currentTask]; currentNode != null; currentNode = currentNode.ThreadNext)
										currentNode.ComputeChoiceAndMatchFunction(x_F1_vec, t_F1_vec, mask, alpha);
								}
							break;
						case ActivationType.Prediction:
								for(long currentTask = 0; currentTask < LibTopoART_info.MaximumThreads; ++currentTask) {
									for(var currentNode = _threadArray[currentTask]; currentNode != null; currentNode = currentNode.ThreadNext)
										currentNode.ComputeAlternativeChoiceFunction(x_F1_vec, mask);
								}
							break;
					}

					return;
				}

				for(long currentTask = 0; currentTask < LibTopoART_info.MaximumThreads; ++currentTask) {
					var currentTaskNode = _threadArray[currentTask];

					// skip empty node lists
					if(currentTaskNode == null) {
						_tasks[currentTask] = Task.CompletedTask;
						continue;
					}

					switch(actType) {
						case ActivationType.Training:
								_tasks[currentTask] = new Task(() => {
									for(var currentNode = currentTaskNode; currentNode != null; currentNode = currentNode.ThreadNext)
										currentNode.ComputeChoiceAndMatchFunction(x_F1_vec, t_F1_vec, mask, alpha);
								});
							break;
						case ActivationType.Prediction:
								_tasks[currentTask] = new Task(() => {
									for(var currentNode = currentTaskNode; currentNode != null; currentNode = currentNode.ThreadNext)
										currentNode.ComputeAlternativeChoiceFunction(x_F1_vec, mask);
								});
							break;
					}

					_tasks[currentTask].Start();
				}

				Task.WaitAll(_tasks);
			}
		}

		protected void StopThreads()
		{
			if(_isActive) {
				_isActive = false;
			}
		}
	}

//**********************************************************************************************************************

#if DEBUG
	/// <summary>Class providing common functions and types.</summary>
	public static class Common
#else
	internal static class Common
#endif
	{
		/// <summary>Instance variable <c>types</c> represents an array of possible integer and
		/// floating point types.</summary>
		public static readonly string[] Types = [
			"sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "decimal"
		];

		public const long TopoART_C_default_nu = 3;
		public const long TopoART_R_default_nu = 10;

#if DEBUG
		public const decimal TopoART_file_format_version_default = 1.00m;
		public static decimal TopoART_file_format_version = TopoART_file_format_version_default;
		private const decimal Episodic_TopoART_file_format_version_default = 0.02m;
		public static decimal Episodic_TopoART_file_format_version = Episodic_TopoART_file_format_version_default;
		public static long t_max_test_v001 = 0;
#else
		public const decimal TopoART_file_format_version = 1.00m;
		public const decimal Episodic_TopoART_file_format_version = 1.00m;
#endif
		public const decimal TopoART_AM_file_format_version = 1.00m;
		public const decimal TopoART_C_file_format_version = 1.00m;
		public const decimal TopoART_R_file_format_version = 1.00m;
		public const decimal Hypersphere_TopoART_file_format_version = 1.00m;
		public const decimal Hypersphere_TopoART_C_file_format_version = 1.00m;

		private const decimal HeaderOffsetBeginModulo	=	0.0000000001m;
		private const decimal HeaderOffsetEndModulo		=	0.000000000001m;

		internal const string InvalidClassIDException_NegativeClassID			=	"Negative class IDs are not allowed";
		internal const string InvalidClassIDException_InvalidClassID			=	"Invalid class ID";

		internal const string InvalidModuleIndexException_OutOfRangeID			=	"Module index out of range";

		internal const string InvalidFileException_FileCorrupted				=	"File corrupted";
		internal const string InvalidFileException_InvalidVersion				=	"Invalid file format version";
		internal const string InvalidFileException_UnsupportedFileType			=	"Unsupported file type";
		internal const string InvalidFileException_UnsupportedFloatType			=	"Unsupported float type";
		internal const string InvalidFileException_UnsupportedIntegerType		=	"Unsupported integer type";
		internal const string InvalidFileException_UnsupportedNetworkType		=	"Unsupported network type";

		internal const string InvalidNumberException_EdgeNumberTooHigh			=	"Too many edges for the operation to be executed";
		internal const string InvalidNumberException_NodeNumberTooHigh			=	"Too many nodes for the operation to be executed";

		internal const string InvalidSizeException_InvalidPhiArrayLength		=	"Phi array length does not equal the module number";

		internal const string InvalidStateException_InvalidNetworkState			=	"Network is in an invalid state";

		internal const string InvalidTypeException_UnsupportedNetworkType		=	"Unsupported network type";

		internal const string Warning_CorrectedInputValue						=	"Too low or too high input value(s) set to the closest permitted value(s)";

		internal const long ScalingFactor = 1000000000;
		internal const long	ScalingFactorByte = (int)(ScalingFactor / 255);
		internal static readonly Vector<int> ScalingVectorInt = new Vector<int>((int)ScalingFactor);
		internal static readonly Vector<long> ScalingVectorLong = new Vector<long>(ScalingFactor);
		internal static readonly Vector<int> ScalingVectorByteInt = new Vector<int>((int)ScalingFactorByte);

//----------------------------------------------------------------------------------------------------------------------

		internal static long ComputeClusterIDs<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>(
			IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? nodes, long phi)
		{
			for(var currentNode = nodes; currentNode != null; currentNode = currentNode.Next)
				currentNode.ClusterID = LibTopoART_info.UNDEFINED;

			long currentID = 1;
			var toBeProcessedQueue = new Queue<IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>>();
			for(var currentNode = nodes; currentNode != null; currentNode = currentNode.Next) {
				if((!currentNode.IsNodeCandidate(phi)) && (currentNode.ClusterID == LibTopoART_info.UNDEFINED)) {
					toBeProcessedQueue.Enqueue(currentNode);
					while(toBeProcessedQueue.Count != 0) {
						var node = toBeProcessedQueue.Dequeue();

						if(node.ClusterID != LibTopoART_info.UNDEFINED)
							continue;

						node.ClusterID = currentID;

						node.GetConnectedNodeIDs(out long size, out long[]? connectedNodeIDs);
						if(connectedNodeIDs != null) {
							for(long i = 0; i < size; ++i) {
								var connectedNode = FindNodeFromID(nodes!, connectedNodeIDs[i]);
								if(connectedNode != null) {
									if((!connectedNode.IsNodeCandidate(phi)) && (connectedNode.ClusterID == LibTopoART_info.UNDEFINED))
										toBeProcessedQueue.Enqueue(connectedNode);
								}
							}
						}
					}
					++currentID;
				}
			}

			var clusterNum = currentID - 1;
			Message($"Found {clusterNum} clusters");
			return clusterNum;
		}

		public static Vector<TType>[]? CreateEncodedVectorArray<TType>(TType[]? arr) where TType : struct
		{
			Vector<TType>[]? vecSimd;

			Debug.Assert((arr == null) || (arr.LongLength % 2 == 0));

			if(arr == null)
				vecSimd = null;
			else {
				long i, j;
				var halfSize = arr.LongLength >> 1;
				var halfSizeSimd = SimdLength<TType>(halfSize);
				vecSimd = new Vector<TType>[halfSizeSimd << 1];

				for(i = 0, j = 0; i < halfSize && (halfSize - i >= Vector<TType>.Count);  i += Vector<TType>.Count, ++j) {
					vecSimd[j] = new Vector<TType>(arr, (int)i);
					vecSimd[j + halfSizeSimd] = new Vector<TType>(arr, (int)(i + halfSize));
				}

				if(i < halfSize) {
					var tmpArr1 = new TType[Vector<TType>.Count];
					var tmpArr2 = new TType[Vector<TType>.Count];

					for(long k = 0; i < halfSize; ++i, ++k) {
						tmpArr1[k] = arr[i];
						tmpArr2[k] = arr[i + halfSize];
					}
					vecSimd[j] = new Vector<TType>(tmpArr1);
					vecSimd[j + halfSizeSimd] = new Vector<TType>(tmpArr2);
				}
			}

			return vecSimd;
		}

		internal static void FillMaskVectorArray(bool[] mask, int[] paddedMaskInt, Vector<int>[] maskSimd)
		{
			Debug.Assert(maskSimd.LongLength == SimdLength<int>(mask.LongLength));
			Debug.Assert(paddedMaskInt.LongLength == maskSimd.LongLength * Vector<int>.Count);

			// create excitatory mask vector (-1 == 0xffffffff)
			for(long i = 0; i < mask.LongLength; ++i)
				paddedMaskInt[i] = mask[i] ? 0 : -1;

			for(long i = 0, j = 0; j < maskSimd.LongLength; i += Vector<int>.Count, ++j)
				maskSimd[j] = new Vector<int>(paddedMaskInt, (int)i);
		}

		internal static void FillMaskVectorArray(long iLen, long dLen, bool[] m_i_vec, int[] paddedMaskInt, Vector<int>[] maskSimd)
		{
			Debug.Assert(maskSimd.LongLength == SimdLength<int>(iLen + dLen));
			Debug.Assert(paddedMaskInt.LongLength == maskSimd.LongLength * Vector<int>.Count);

			// create excitatory mask vector (-1 == 0xffffffff)
			for(long i = 0; i < iLen; ++i)
				paddedMaskInt[i] = m_i_vec[i] ? 0 : -1;
			for(long i = 0; i < dLen; ++i)
				paddedMaskInt[i + iLen] = 0;

			for(long i = 0, j = 0; j < maskSimd.LongLength; i += Vector<int>.Count, ++j)
				maskSimd[j] = new Vector<int>(paddedMaskInt, (int)i);
		}

		internal static Vector<TType>[] CreateVectorArray<TType>(TType[] arr) where TType : struct
		{
			long i, j;

			var vecSimd = new Vector<TType>[SimdLength<TType>(arr.LongLength)];
			for(i = 0, j = 0; (i < arr.LongLength) && (arr.LongLength - i >= Vector<TType>.Count);  i += Vector<TType>.Count, ++j)
				vecSimd[j] = new Vector<TType>(arr, (int)i);

			if(i < arr.LongLength) {
				var tmpArr = new TType[Vector<TType>.Count];
				for(long k = 0; i < arr.LongLength; ++i, ++k)
					tmpArr[k] = arr[i];
				vecSimd[j] = new Vector<TType>(tmpArr);
			}

			return vecSimd;
		}

		private static IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>?
			FindNodeFromID<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>
			(IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? nodes, long nodeID)
		{
			for(var currentNode = nodes; currentNode != null; currentNode = currentNode.Next) {
				if(currentNode.NodeID == nodeID)
					return currentNode;
			}

			return null;
		}

		internal static List<CategoryInfo>? GetCategoryInfos<TFloatType, TSpatialWeightType, TTemporalWeightType,
			TMaskType>(long nodeNum, IF2_node<TFloatType, TSpatialWeightType, TTemporalWeightType, TMaskType>? nodes)
		{
			if(nodeNum > int.MaxValue)
				throw new InvalidNumberException(InvalidNumberException_NodeNumberTooHigh);
			else if(nodeNum <= 0)
				return null;

			List<CategoryInfo> list = new List<CategoryInfo>((int)nodeNum);

			for(var currentNode = nodes; currentNode != null; currentNode = currentNode.Next)
				list.Add(new CategoryInfo(currentNode.GetCopyOfSpatialWeights(), currentNode.GetCopyOfTemporalWeights(),
					currentNode.ClusterID, currentNode.ClassID));

			return list;
		}

		internal static void InitialisationMessage(string networkName)
		{
			Message("Create " + networkName + " network");
		}

		internal static void InitTopoARTRMasks(out bool[] trainMask, out bool[] default_m_i_vec, long iLen, long dLen)
		{
			//allow for separated match functions for i and d
			//attention: in the original algorithm realised by index sets
			trainMask = new bool[iLen + dLen];
			for(long i = 0; i < iLen; ++i)
				trainMask[i] = false;
			for(long i = 0; i < dLen; ++i)
				trainMask[i + iLen] = true;

			default_m_i_vec = new bool[iLen];
			for(long i = 0; i < iLen; ++i)
				default_m_i_vec[i] = false;
		}

		internal static void InitTopoARTRMasks(out int[] trainMask, out int[] defaultMask, long iLen, long dLen)
		{
			//allow for separated match functions for i and d
			//attention: in the original algorithm realised by index sets
			// excitatory mask vector (-1 == 0xffffffff)
			trainMask = new int[iLen + dLen];
			for(long i = 0; i < iLen; ++i)
				trainMask[i] = -1;
			for(long i = 0; i < dLen; ++i)
				trainMask[i + iLen] = 0;

			// excitatory mask vector (-1 == 0xffffffff)
			defaultMask = new int[iLen + dLen];
			for(long i = 0; i < iLen; ++i)
				defaultMask[i] = -1;
			for(long i = 0; i < dLen; ++i)
				defaultMask[i + iLen] = 0;
		}

		public static HeaderInfo LoadBinaryHeader(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			using var reader = new BinaryReader(file);
			return LoadBinaryHeader(reader, null);
		}

		public static HeaderInfo LoadBinaryHeader(Stream stream)
		{
			using var reader = new BinaryReader(stream, Encoding.UTF8, true);
			return LoadBinaryHeader(reader, null);
		}

		public static HeaderInfo LoadBinaryHeader(BinaryReader reader, NetworkType networkType,
			string networkName, in decimal referenceFileFormatVersion, TypeIndex integerType,
			TypeIndex floatType)
		{
			return LoadBinaryHeader(reader, networkType, networkName, new FileFormatVersions(referenceFileFormatVersion),
									integerType, floatType);
		}

		public static HeaderInfo LoadBinaryHeader(BinaryReader reader, NetworkType networkType,
			string networkName, in FileFormatVersions referenceFileFormatVersions, TypeIndex integerType,
			TypeIndex floatType)
		{
			string warningPrefix;

			var info = LoadBinaryHeader(reader, referenceFileFormatVersions);

			switch(referenceFileFormatVersions.HeaderOffset) {
				case 1:
						Message(string.Format(CultureInfo.InvariantCulture,
												"Initialise {0} network from file (version = {1:N2}; file format version = {2:N2})",
												networkName, info.Version, info.FileFormatVersions.FileFormatVersion));
						warningPrefix = "File";
					break;
				case 2:
						Message(string.Format(CultureInfo.InvariantCulture,
												"Initialise {0} network from file (version = {1:N2}; file format version = {2:N2}; TopoART file format version = {3:N2})",
												networkName, info.Version, info.FileFormatVersions.FileFormatVersion, info.FileFormatVersions.TopoARTFileFormatVersion));
						if(info.FileFormatVersions.FileFormatVersion != referenceFileFormatVersions.FileFormatVersion)
							Warning("File format version does not match");
						warningPrefix = "TopoART file";
					break;
				case 3:
						Message(string.Format(CultureInfo.InvariantCulture,
												"Initialise {0} network from file (version = {1:N2}; file format version = {2:N2}; base file format version = {3:N2}); " +
												"TopoART file format version = {4:N2})", networkName, info.Version, info.FileFormatVersions.FileFormatVersion,
												info.FileFormatVersions.BaseFileFormatVersion, info.FileFormatVersions.TopoARTFileFormatVersion));
						if(info.FileFormatVersions.FileFormatVersion != referenceFileFormatVersions.FileFormatVersion)
							Warning("File format version does not match");
						if(info.FileFormatVersions.BaseFileFormatVersion != referenceFileFormatVersions.BaseFileFormatVersion)
							Warning("Base file format version does not match");
						warningPrefix = "TopoART file";
					break;
				default:
					throw new InvalidFileException(InvalidFileException_UnsupportedFileType);
			}

			if(info.FileFormatVersions.TopoARTFileFormatVersion >= 0.09m) {
				if(info.FileFormatVersions.TopoARTFileFormatVersion != referenceFileFormatVersions.TopoARTFileFormatVersion)
					Warning(warningPrefix + " format version does not match");

				if(info.FileFormatVersions.TopoARTFileFormatVersion >= 0.10m) {
					info.CompatibilityMode(integerType, floatType);

					if(info.Type != networkType)
						throw new InvalidFileException(InvalidFileException_UnsupportedNetworkType);
				}
			}

			return info;
		}

		public static void SaveBinaryHeader(BinaryWriter writer, NetworkType networkType, in decimal taFileFormatVersion,
			(TypeIndex, TypeIndex) typeIndexes, CompressionLevel compression)
		{
			SaveBinaryHeader(writer, networkType, new FileFormatVersions(taFileFormatVersion), typeIndexes, compression);
		}

		public static void SaveBinaryHeader(BinaryWriter writer, NetworkType networkType, in FileFormatVersions fileFormatVersions,
			(TypeIndex, TypeIndex) typeIndexes, CompressionLevel compression)
		{
			writer.Write(LibTopoART_info.version);

			var unit = (fileFormatVersions.TopoARTFileFormatVersion >= 0.10m) ? Common.HeaderOffsetEndModulo : 0.0m;

			Debug.Assert((fileFormatVersions.HeaderOffset == 1) || (fileFormatVersions.HeaderOffset == 2) || (fileFormatVersions.HeaderOffset == 3));

			// write specific file format versions only if required and signify header size
			switch(fileFormatVersions.HeaderOffset) {
				case 1:
						writer.Write(fileFormatVersions.TopoARTFileFormatVersion + unit);
					break;
				case 2:
						writer.Write(fileFormatVersions.FileFormatVersion + unit * 2.0m);
						writer.Write(fileFormatVersions.TopoARTFileFormatVersion);
					break;
				case 3:
						writer.Write(fileFormatVersions.FileFormatVersion + unit * 3.0m);
						writer.Write(fileFormatVersions.BaseFileFormatVersion);
						writer.Write(fileFormatVersions.TopoARTFileFormatVersion);
					break;
			}

			if(fileFormatVersions.TopoARTFileFormatVersion >= 0.10m) {
				writer.Write((uint)typeIndexes.Item1);
				writer.Write((uint)typeIndexes.Item2);
				writer.Write((uint)networkType);
				writer.Write((uint)(compression == CompressionLevel.NoCompression ? SaveFlags.None : SaveFlags.GZip));
			}
		}

		internal static void SaveCompatibleBinaryHeader(BinaryWriter writer, NetworkType networkType,
			in decimal taFileFormatVersion, CompressionLevel compression)
		{
			SaveCompatibleBinaryHeader(writer, networkType, new FileFormatVersions(taFileFormatVersion), compression);
		}

		internal static void SaveCompatibleBinaryHeader(BinaryWriter writer, NetworkType networkType,
			in FileFormatVersions fileFormatVersions, CompressionLevel compression)
		{
			SaveBinaryHeader(writer, networkType, fileFormatVersions, (TypeIndex.LongIndex, TypeIndex.DecimalIndex), compression);
		}

		internal static long SimdLength<TType>(long n) where TType : struct {
			return ((n - 1) / Vector<TType>.Count + 1);
		}

		internal static long SimdLengthEncoded<TType>(long n) where TType : struct {
			return 2 * SimdLength<TType>(n >> 1);
		}

		public static (long, int) SimdIndexes<TType>(long n, long nEnc) where TType : struct
		{
			if(n < nEnc)
				return ((n / Vector<TType>.Count), (int)(n % Vector<TType>.Count));
			else {
				long nTmp = n - nEnc;
				return ((nTmp / Vector<TType>.Count) + SimdLength<TType>(nEnc), (int)(nTmp % Vector<TType>.Count));
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static long HorizontalSum(Vector<long> v)
		{
#if NET6_0_OR_GREATER
			return Vector.Sum(v);
#else
			long sum = 0;
			for(var j = 0; j < Vector<long>.Count; ++j)
				sum += v[j];
			return sum;
#endif
		}

		internal static void Message(string message, VerbosityLevel verbosity = VerbosityLevel.Normal)
		{
			if((uint)verbosity <= (uint)LibTopoART_control.verbosity)
				Console.WriteLine(message);
		}

		internal static void Warning(string warning, VerbosityLevel verbosity = VerbosityLevel.Normal)
		{
			if((uint)verbosity <= (uint)LibTopoART_control.verbosity)
				Console.WriteLine("Warning: " + warning);
		}

//----------------------------------------------------------------------------------------------------------------------

		private static HeaderInfo LoadBinaryHeader(BinaryReader reader, in FileFormatVersions? referenceFileFormatVersions)
		{
			var version = reader.ReadDecimal();
			var extendedFileFormatVersion = reader.ReadDecimal();
			var tmpModulo = extendedFileFormatVersion % HeaderOffsetBeginModulo;
			var fileFormatVersion = extendedFileFormatVersion - tmpModulo;
			long headerOffset = (ushort)decimal.Round((tmpModulo - (tmpModulo % HeaderOffsetEndModulo)) / HeaderOffsetEndModulo);

			var cmpHeaderOffset = headerOffset;
			if(referenceFileFormatVersions is FileFormatVersions f) {
				cmpHeaderOffset = f.HeaderOffset;

				if((headerOffset != 0) && (headerOffset != f.HeaderOffset))
					throw new InvalidFileException(InvalidFileException_UnsupportedFileType);
			} else if(headerOffset == 0)
				// Only files of TA version 0.10m or higher contain the header offset.
				throw new InvalidFileException(InvalidFileException_InvalidVersion);

			decimal taFileFormatVersion;
			var baseFileFormatVersion = 0.0m;
			switch(cmpHeaderOffset) {
				case 1:
						taFileFormatVersion = fileFormatVersion;
					break;
				case 2:
						taFileFormatVersion = reader.ReadDecimal();
					break;
				case 3:
						baseFileFormatVersion = reader.ReadDecimal();
						taFileFormatVersion = reader.ReadDecimal();
					break;
				default:
					throw new InvalidFileException(InvalidFileException_UnsupportedFileType);
			}

			NetworkType type = NetworkType.NotSet;
			SaveFlags flags = SaveFlags.None;
			TypeIndex? intType = null;
			TypeIndex? floatType = null;
			if(taFileFormatVersion < 0.09m)
				throw new InvalidFileException(InvalidFileException_InvalidVersion);
			else {
				if(taFileFormatVersion >= 0.10m) {
					intType = (TypeIndex)reader.ReadUInt32();
					floatType = (TypeIndex)reader.ReadUInt32();
					type = (NetworkType)reader.ReadUInt32();

					uint readFlags = reader.ReadUInt32();
					if(readFlags != (uint)SaveFlags.None && readFlags != (uint)SaveFlags.GZip)
						throw new InvalidFileException(InvalidFileException_UnsupportedFileType);
					flags = (SaveFlags)readFlags;
				}
			}

			switch(cmpHeaderOffset) {
				case 1:
					return (new HeaderInfo(version, type, new FileFormatVersions(fileFormatVersion), flags, intType, floatType));
				case 2:
					return (new HeaderInfo(version, type, new FileFormatVersions(fileFormatVersion, taFileFormatVersion), flags, intType, floatType));
				case 3:
					return (new HeaderInfo(version, type, new FileFormatVersions(fileFormatVersion, baseFileFormatVersion, taFileFormatVersion), flags, intType, floatType));
				default:
					throw new InvalidFileException(InvalidFileException_UnsupportedFileType);
			}
		}
	}

//**********************************************************************************************************************

#pragma warning restore 1591
}