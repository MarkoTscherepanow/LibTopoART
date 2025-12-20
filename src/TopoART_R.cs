/*"*********************************************************************************************************************
*                                                   TopoART-R classes                                                  *
*                                     created by Marko Tscherepanow, 6 August 2011                                     *
************************************************************************************************************************
*                                  $Id: TopoART_R.cs 1680 2025-11-15 17:23:15Z marko $                                 *
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

	/// <summary>Class <c>TopoART_R</c> provides an implementation of the TopoART-R neural network as proposed in
	/// "Marko Tscherepanow (2011). An Extended TopoART Network for the Stable On-Line Learning of Regression Functions.
	/// In Proceedings of the International Conference on Neural Information Processing (ICONIP), LNCS 7063, pp.
	/// 562–571. Berlin, Germany: Springer."
	/// <para>Class <c>TopoART_R</c> requires all input and output to lie in the interval [0, 1].</para>
	/// </summary>
	public class TopoART_R : TopoART, ITopoART_R
	{
		private bool[]? _trainMask;
		private bool[]? _default_m_i_vec;
		private const string _networkName = "TopoART-R";
		private const NetworkType _networkType = NetworkType.TopoARTR;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>D_len</c> returns the length of the output vector (dependent variables).</summary>
		public long D_len { get; private set; }

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>TopoART_R</c>.</summary>
		public new decimal FileFormatVersion { get => Common.TopoART_R_file_format_version; }

		/// <summary>Property <c>I_len</c> returns the length of the input vector (independent variables).</summary>
		public long I_len { get; private set; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the
		/// neighbourhood set N during prediction. If the parameter <c>nu</c> is not explicitly provided for prediction,
		/// this property will be applied. (This parameter does not modify the network. It may be arbitrarily changed
		/// for each prediction step.)</summary>
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
		} = Common.TopoART_R_default_nu;

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

		/// <summary>This constructor initialises a TopoART-R network.</summary>
		/// <param name="iLen">The length of the input vector (independent variables) to be learnt.</param>
		/// <param name="dLen">The length of the output vector (dependent variables) to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART-R modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART-R module (TopoART-R a).</param>
		public TopoART_R(long iLen, long dLen, long moduleNum, decimal rho_a)
			: base(CheckLength(iLen) + CheckLength(dLen), moduleNum, rho_a) 
		{
			I_len = CheckLength(iLen);
			if(I_len != iLen)
				Common.Warning("Invalid length of vector i, changed to " + I_len);

			D_len = CheckLength(dLen);
			if(D_len != dLen)
				Common.Warning("Invalid length of vector d, changed to " + D_len);

			InitTransientMembers();
		}

		/// <summary>This constructor loads a saved TopoART-R network.</summary>
		/// <param name="path">The path of a binary TopoART-R file.</param>
		/// <exception cref="InvalidFileException">Throws when the given file cannot be loaded.</exception>
		public TopoART_R(string path) : base(path)
		{
			InitTransientMembers();
		}

		private void InitTransientMembers()
		{
			Common.InitTopoARTRMasks(out _trainMask, out _default_m_i_vec, I_len, D_len);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step. The independent variables and the dependent variables
		/// are automatically separated.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			Debug.Assert(_trainMask != null);
			LearnWithMask(input, _trainMask);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector (independent variables) to be learnt.</param>
		/// <param name="output">The output vector (dependent variables) corresponding to <paramref name="input"/>.
		/// </param>
		public void Learn(decimal[] input, decimal[] output)
		{
			Debug.Assert(input.LongLength == I_len);
			Debug.Assert(output.LongLength == D_len);
			Debug.Assert(_trainMask != null);
			Debug.Assert(_x_F0 != null);

			input.CopyTo(_x_F0, 0);
			output.CopyTo(_x_F0, I_len);
			LearnWithMask(_x_F0!, _trainMask!);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method predicts the dependent variables using the default value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		public decimal[] Predict(decimal[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the dependent variables using a custom value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		public decimal[] Predict(decimal[] input, long nu)
		{
			Debug.Assert(_default_m_i_vec != null);
			return Predict(input, _default_m_i_vec!, nu).d_vec_prediction;
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables using the
		/// default value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		public TopoART_R_prediction<decimal> Predict(decimal[] input, bool[] mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables using a
		/// custom value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not alter the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		public TopoART_R_prediction<decimal> Predict(decimal[] input, bool[] mask, long nu)
		{
			Debug.Assert(I_len == input.LongLength);
			Debug.Assert(I_len == mask.LongLength);

			lock(_learningLock) {
				CompleteLearningQueue();

				Debug.Assert(_modules != null);
				Debug.Assert(_x_F0 != null);

				if(_modules![ModuleNum - 1]._nodeNum < 1) 
					return new TopoART_R_prediction<decimal>();	//empty net

				if(nu < 1) {
					nu = 1;
					Common.Warning("Invalid value for nu, changed to " + nu);
				}

				var tmpMask = new bool[_x_F0_len];

				for(long i = 0; i < I_len; ++i) {			// concatenate vectors
					if(mask[i] == false)
						_x_F0![i] = input[i];
					else
						_x_F0![i] = 0.0m;

					tmpMask[i] = mask[i];
				}

				for(long i = 0; i < D_len; ++i) {
					_x_F0![i + I_len] = 0.0m;
					tmpMask[i + I_len] = true;
				}

				var x_F1 = EncodeCurrentInput();

				var tauVec = new decimal[_x_F0_len * 2];
				for(long i = 0; i < _x_F0_len * 2; ++i) 
					tauVec[i] = 0.0m;

				if(_modules![ModuleNum - 1].ComputeAlternativeChoiceFunctionsWithMaskAndNu(
					x_F1, tmpMask, nu, out Stack<TA_F2_node> enclosingNodes, out List<TA_F2_node> neighbouringNodes)) {
					if(enclosingNodes.Count > 0) {
						do											// add at least one node
						{
							var currentNode = enclosingNodes.Pop();
							var currentWeights = currentNode.Weights;
							for(long i = 0; i < _x_F0_len * 2; ++i)
								tauVec[i] = Math.Max(tauVec[i], currentWeights[i]);
						} while(enclosingNodes.Count > 0);
					} else if(neighbouringNodes.Count > 0) {
						var invSum = 0.0m;
						foreach(var node in neighbouringNodes)
							invSum += 1.0m / (1.0m - node.Activation);

						foreach(TA_F2_node node in neighbouringNodes) {
							var currentWeights = node.Weights;
							for(long i = 0; i < _x_F0_len * 2; ++i)
								tauVec[i] += ((1.0m / (1.0m - node.Activation) * currentWeights[i]) / invSum);
						}
					} 
				} else
					return new TopoART_R_prediction<decimal>();

				var iVecPrediction = new decimal[I_len];
				var dVecPrediction = new decimal[D_len];
				var prediction = new TopoART_R_prediction<decimal>(iVecPrediction, dVecPrediction);
				for(long i = 0; i < I_len; ++i) {
					if(mask[i] == false)
						iVecPrediction[i] = prediction.NO_PREDICTION;
					else
						iVecPrediction[i] = 0.5m * tauVec[i] + 0.5m * (1.0m - tauVec[I_len + D_len + i]);
				}
				for(long i = 0; i < D_len; ++i)
					dVecPrediction[i] = 0.5m * tauVec[I_len + i] + 0.5m * (1.0m - tauVec[2 * I_len + D_len + i]);

				return prediction;
			}
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override (FileFormatVersions, SaveFlags) LoadBinaryHeader(BinaryReader reader)
		{
			var headerInfo = Common.LoadBinaryHeader(reader, _networkType, _networkName, new FileFormatVersions(FileFormatVersion,
								TopoARTFileFormatVersion), integerType, floatType);
			return (headerInfo.FileFormatVersions, headerInfo.Flags);
		}

		private protected override void LoadPrecedingBinaryInformation(BinaryReader reader, FileFormatVersions fileFormatVersions)
		{
			base.LoadPrecedingBinaryInformation(reader, fileFormatVersions);

			I_len = reader.ReadInt64();
			D_len = reader.ReadInt64();

			if(fileFormatVersions.FileFormatVersion >= 1.0m)
				Nu = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*       TopoART-R network       *");
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
			writer.WriteLine("i length: " + I_len);
			writer.WriteLine("d length: " + D_len);
			writer.WriteLine("nu: " + Nu);
			writer.WriteLine("*********************************");
		}

		private protected override void SaveBinaryHeader(BinaryWriter writer, CompressionLevel compression)
		{
			Common.SaveBinaryHeader(writer, _networkType, new FileFormatVersions(FileFormatVersion, TopoARTFileFormatVersion),
				(integerType, floatType), compression);
		}

		private protected override void SavePrecedingBinaryInformation(BinaryWriter writer)
		{
			base.SavePrecedingBinaryInformation(writer);
			writer.Write(I_len);
			writer.Write(D_len);
			writer.Write(Nu);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}
	}

//**********************************************************************************************************************

	/// <summary>Class <c>Fast_TopoART_R</c> provides an implementation of the TopoART-R neural network as proposed in
	/// "Marko Tscherepanow (2011). An Extended TopoART Network for the Stable On-Line Learning of Regression Functions.
	/// In Proceedings of the International Conference on Neural Information Processing (ICONIP), LNCS 7063, pp.
	/// 562–571. Berlin, Germany: Springer."
	/// <para>Internally, real-valued data are mapped to <c>int</c> variables. Therefore, computations are accelerated
	/// but less accurate. As a consequence, the results may differ slightly from class <c>TopoART_R</c>.</para>
	/// <para>Class <c>Fast_TopoART_R</c> requires all input and output to lie in the interval [0, 1].</para>
	/// </summary>
	public class Fast_TopoART_R : Fast_TopoART, IFast_TopoART_R
	{
		private Vector<int>[]? _trainMask;
		private Vector<int>[]? _defaultMask;
		private long _nu = Common.TopoART_R_default_nu;
		private const string _networkName = "TopoART-R";
		private const NetworkType _networkType = NetworkType.TopoARTR;

		private decimal[]? _x_F0_decimal;
		private byte[]? _x_F0_byte;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>D_len</c> returns the length of the output vector (dependent variables).</summary>
		public long D_len { get; private set; }

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Fast_TopoART_R</c>.</summary>
		public new decimal FileFormatVersion { get => Common.TopoART_R_file_format_version; }

		/// <summary>Property <c>I_len</c> returns the length of the input vector (independent variables).</summary>
		public long I_len { get; private set; }

		/// <summary>Property <c>Nu</c> represents the default value used for the maximum cardinality of the
		/// neighbourhood set N during prediction. If the parameter <c>nu</c> is not explicitly provided for prediction,
		/// this property will be applied. (This parameter does not modify the network. It may be arbitrarily changed
		/// for each prediction step.)</summary>
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

		/// <summary>This constructor initialises a TopoART-R network.</summary>
		/// <param name="iLen">The length of the input vector (independent variables) to be learnt.</param>
		/// <param name="dLen">The length of the output vector (dependent variables) to be learnt.</param>
		/// <param name="moduleNum">The number of TopoART-R modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first TopoART-R module (TopoART-R a).</param>
		public Fast_TopoART_R(long iLen, long dLen, long moduleNum, decimal rho_a)
			: base(CheckLength(iLen) + CheckLength(dLen), moduleNum, rho_a) 
		{
			I_len = CheckLength(iLen);
			if(I_len != iLen)
				Common.Warning("Invalid length of vector i, changed to " + I_len);

			D_len = CheckLength(dLen);
			if(D_len != dLen)
				Common.Warning("Invalid length of vector d, changed to " + D_len);

			InitTransientMembers();
		}

		/// <summary>This constructor loads a saved TopoART-R network.</summary>
		/// <param name="path">The path of a binary TopoART-R file.</param>
		/// <exception cref="InvalidFileException">Throws when the given file cannot be loaded.</exception>
		public Fast_TopoART_R(string path) : base(path)
		{
			InitTransientMembers();
		}

		private void InitTransientMembers()
		{
			Common.InitTopoARTRMasks(out int[] tmpTrainMask, out int[] tmpDefaultMask, I_len, D_len);
			_trainMask = Common.CreateVectorArray(tmpTrainMask);
			_defaultMask = Common.CreateVectorArray(tmpDefaultMask);
			_x_F0_decimal = new decimal[I_len + D_len];
			_x_F0_byte = new byte[I_len + D_len];
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method performs a single training step. The independent variables and the dependent variables
		/// are automatically separated.</summary>
		/// <param name="input">The input vector to be learnt. The input values are internally scaled from [0, 255] to
		/// [0, 1].</param>
		public override void Learn(byte[] input)
		{
			Debug.Assert(input.LongLength == I_len + D_len);
			Debug.Assert(_trainMask != null);
			LearnWithMask(input, _trainMask!);
		}

		/// <summary>This method performs a single training step. The independent variables and the dependent variables
		/// are automatically separated.</summary>
		/// <param name="input">The input vector to be learnt.</param>
		public override void Learn(decimal[] input)
		{
			Debug.Assert(input.LongLength == I_len + D_len);
			Debug.Assert(_trainMask != null);
			LearnWithMask(input, _trainMask!);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector (independent variables) to be learnt. The elements of the input vector
		/// are internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="output">The output vector (dependent variables) corresponding to <paramref name="input"/>. The
		/// elements of the output vector are internally scaled from [0, 255] to [0, 1].</param>
		public void Learn(byte[] input, byte[] output)
		{
			Debug.Assert(input.LongLength == I_len);
			Debug.Assert(output.LongLength == D_len);
			Debug.Assert(_trainMask != null);
			Debug.Assert(_x_F0_byte != null);

			input.CopyTo(_x_F0_byte, 0);
			output.CopyTo(_x_F0_byte, I_len);
			LearnWithMask(_x_F0_byte!, _trainMask!);
		}

		/// <summary>This method performs a single training step.</summary>
		/// <param name="input">The input vector (independent variables) to be learnt.</param>
		/// <param name="output">The output vector (dependent variables) corresponding to <paramref name="input"/>.
		/// </param>
		public void Learn(decimal[] input, decimal[] output)
		{
			Debug.Assert(input.LongLength == I_len);
			Debug.Assert(output.LongLength == D_len);
			Debug.Assert(_trainMask != null);
			Debug.Assert(_x_F0_decimal != null);

			input.CopyTo(_x_F0_decimal, 0);
			output.CopyTo(_x_F0_decimal, I_len);
			LearnWithMask(_x_F0_decimal!, _trainMask!);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>This method predicts the dependent variables using the default value of nu.</summary>
		/// <param name="input">The input vector (independent variables). The elements of the input vector are
		/// internally scaled from [0, 255] to [0, 1].</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		public byte[] Predict(byte[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the dependent variables using the default value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		public decimal[] Predict(decimal[] input)
		{
			return Predict(input, Nu);
		}

		/// <summary>This method predicts the dependent variables using a custom value of nu.</summary>
		/// <param name="input">The input vector (independent variables). The elements of the input vector are
		/// internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted values for all dependent variables. The elements of the predicted output vector are
		/// internally scaled from [0, 1] to [0, 255].</returns>
		public byte[] Predict(byte[] input, long nu)
		{
			Debug.Assert(I_len == input.LongLength);
			Debug.Assert(_defaultMask != null);
			Debug.Assert(Common.SimdLength<int>(I_len + D_len) == _defaultMask!.LongLength, "length mismatch: " +
						(I_len + D_len).ToString() + " != " + _defaultMask!.LongLength.ToString());

			return PredictInternal(input, _defaultMask!, nu).d_vec_prediction;
		}

		/// <summary>This method predicts the dependent variables using a custom value of nu.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns>The predicted values for all dependent variables.</returns>
		public decimal[] Predict(decimal[] input, long nu)
		{
			Debug.Assert(I_len == input.LongLength);
			Debug.Assert(_defaultMask != null);
			Debug.Assert(Common.SimdLength<int>(I_len + D_len) == _defaultMask!.LongLength, "length mismatch: " +
						(I_len + D_len).ToString() + " != " + _defaultMask!.LongLength.ToString());
			
			return PredictInternal(input, _defaultMask!, nu).d_vec_prediction;
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables using the
		/// default value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables). The elements of the input vector are
		/// internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		public TopoART_R_prediction<byte> Predict(byte[] input, bool[] mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables using the
		/// default value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		public TopoART_R_prediction<decimal> Predict(decimal[] input, bool[] mask)
		{
			return Predict(input, mask, Nu);
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables using a
		/// custom value of nu. Unknown values of independent variables can be signified by setting the corresponding
		/// value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables). The elements of the input vector are
		/// internally scaled from [0, 255] to [0, 1].</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not modify the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables. The elements of the predicted vectors are internally
		/// scaled from [0, 1] to [0, 255].</returns>
		public TopoART_R_prediction<byte> Predict(byte[] input, bool[] mask, long nu)
		{
			Debug.Assert(I_len == input.LongLength);
			Debug.Assert(I_len == mask.LongLength);
			var maskSimd = Common.CreateMaskVectorArray(I_len, D_len, mask);
			return PredictInternal(input, maskSimd, nu);
		}

		/// <summary>This method predicts the dependent variables for a given set of independent variables
		/// using a custom value of nu. Unknown values of independent variables can be signified by setting
		/// the corresponding value of <paramref name="mask"/> to <c>true</c>.</summary>
		/// <param name="input">The input vector (independent variables).</param>
		/// <param name="mask">The mask vector corresponding to <paramref name="input"/>.</param>
		/// <param name="nu">The maximum cardinality of the neighbourhood set N. (In the original TopoART-R network, nu
		/// is fixed to 10. But task-specific adaptations might lead to an improved prediction accuracy. This parameter
		/// does not alter the network. It may be arbitrarily changed in each prediction step.)</param>
		/// <returns> An object of type <c>TopoART_R_prediction</c> containing the predicted values for the unknown
		/// independent variables and all dependent variables.</returns>
		public TopoART_R_prediction<decimal> Predict(decimal[] input, bool[] mask, long nu)
		{
			Debug.Assert(I_len == input.LongLength);
			Debug.Assert(I_len == mask.LongLength);
			var maskSimd = Common.CreateMaskVectorArray(I_len, D_len, mask);
			return PredictInternal(input, maskSimd, nu);
		}

		private TopoART_R_prediction<byte> PredictInternal(byte[] iVec, Vector<int>[] mask, long nu)
		{
			lock(_learningLock) {
				CompleteLearningQueue();

				var tauVec = ComputeTau(() => {
									Debug.Assert(_x_F0_byte != null);
									iVec.CopyTo(_x_F0_byte, 0);
									EncodeCurrentInputSimd(_x_F0_byte!);
								}, mask, nu);

				if(tauVec == null)
					return new TopoART_R_prediction<byte>();

				var dSimd = Common.SimdLength<int>(I_len + D_len);
				var iVecPrediction = new byte[I_len];
				var dVecPrediction = new byte[D_len];
				var prediction = new TopoART_R_prediction<byte>(iVecPrediction, dVecPrediction);

				long i, i1;
				int i2;
				for(i = 0, i1 = 0, i2 = 0; i < I_len; ++i) {
					if(mask[i1][i2] == -1)
						iVecPrediction[i] = prediction.NO_PREDICTION;
					else
						iVecPrediction[i] = ConvertLongToByte((tauVec[i1][i2] + (Common.ScalingFactor - tauVec[i1 + dSimd][i2])) >> 1);

					if(++i2 == Vector<int>.Count) {
						i2 = 0;
						++i1;
					}
				}
				for(i = 0; i < D_len; ++i) {
					dVecPrediction[i] = ConvertLongToByte((tauVec[i1][i2] + (Common.ScalingFactor - tauVec[i1 + dSimd][i2])) >> 1);

					if(++i2 == Vector<int>.Count) {
						i2 = 0;
						++i1;
					}
				}

				return prediction;
			}
		}

		private TopoART_R_prediction<decimal> PredictInternal(decimal[] iVec, Vector<int>[] mask, long nu) {
			lock(_learningLock) {
				CompleteLearningQueue();

				var tauVec = ComputeTau(() => {
									Debug.Assert(_x_F0_decimal != null);
									iVec.CopyTo(_x_F0_decimal, 0);
									EncodeCurrentInputSimd(_x_F0_decimal!);
								}, mask, nu);

				if(tauVec == null)
					return new TopoART_R_prediction<decimal>();

				var dSimd = Common.SimdLength<int>(I_len + D_len);
				var iVecPrediction = new decimal[I_len];
				var dVecPrediction = new decimal[D_len];
				var prediction = new TopoART_R_prediction<decimal>(iVecPrediction, dVecPrediction);

				long i, i1;
				int i2;
				for(i = 0, i1 = 0, i2 = 0; i < I_len; ++i) {
					if(mask[i1][i2] == -1)
						iVecPrediction[i] = prediction.NO_PREDICTION;
					else
						iVecPrediction[i] = ConvertLongToDecimal((tauVec[i1][i2] + (Common.ScalingFactor - tauVec[i1 + dSimd][i2])) >> 1);

					if(++i2 == Vector<int>.Count) {
						i2 = 0;
						++i1;
					}
				}

				for(i = 0; i < D_len; ++i) {
					dVecPrediction[i] = ConvertLongToDecimal((tauVec[i1][i2] + (Common.ScalingFactor - tauVec[i1 + dSimd][i2])) >> 1);

					if(++i2 == Vector<int>.Count) {
						i2 = 0;
						++i1;
					}
				}

				return prediction;
			}
		}

		private Vector<int>[]? ComputeTau(Action encode, Vector<int>[] mask, long nu)
		{
			Debug.Assert(_modules != null);

			if(_modules![ModuleNum - 1]._nodeNum < 1) 
				return null;	//empty net

			if(nu < 1) {
				nu = 1;
				Common.Warning("Invalid value for nu, changed to " + nu);
			}

			Debug.Assert(_x_F1_simd != null);

			encode();

			var x_F1_len_simd = _x_F1_simd!.LongLength;
			var tauVec = new Vector<int>[x_F1_len_simd];

			if(_modules![ModuleNum - 1].ComputeAlternativeChoiceFunctionsWithMaskAndNu(
				_x_F1_simd!, mask, nu, out Stack<FTA_F2_node> enclosingNodes, out List<FTA_F2_node> neighbouringNodes)) {
				if(enclosingNodes.Count > 0) {
					do											// add at least one node
					{
						var currentNode = enclosingNodes.Pop();
						var currentWeights = currentNode.Weights;
						for(long i = 0; i < x_F1_len_simd; ++i)
							tauVec[i] = Vector.Max(tauVec[i], currentWeights[i]);
					} while(enclosingNodes.Count > 0);
				} else if(neighbouringNodes.Count > 0) {
					var invSum = 0.0;
					foreach(var node in neighbouringNodes)
						invSum += Common.ScalingFactor / (double)(Common.ScalingFactor - node.Activation);

					foreach(var node in neighbouringNodes) {
						var frac = Common.ScalingFactor / (double)(Common.ScalingFactor - node.Activation);
						var currentWeights = node.Weights;
						for(long i = 0; i < x_F1_len_simd; ++i) {
							Vector.Widen(currentWeights[i], out Vector<long> tmp1, out Vector<long> tmp2);
							Vector<double> tmp1d = Vector.Divide(frac * Vector.ConvertToDouble(tmp1), Vector<double>.One * invSum);
							Vector<double> tmp2d = Vector.Divide(frac * Vector.ConvertToDouble(tmp2), Vector<double>.One * invSum);
							tauVec[i] += Vector.Narrow(Vector.ConvertToInt64(tmp1d), Vector.ConvertToInt64(tmp2d));
						}
					}
				}
				return tauVec;
			}

			return null;
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

			I_len = reader.ReadInt64();
			D_len = reader.ReadInt64();

			if(fileFormatInfo.Item1.FileFormatVersion >= 1.0m)
				Nu = reader.ReadInt64();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*       TopoART-R network       *");
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
			writer.WriteLine("i length: " + I_len);
			writer.WriteLine("d length: " + D_len);
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
			writer.Write(I_len);
			writer.Write(D_len);
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