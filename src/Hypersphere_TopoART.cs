/*"*********************************************************************************************************************
*                                               Hypersphere TopoART class                                              *
*                                    created by Marko Tscherepanow, 16 February 2014                                   *
************************************************************************************************************************
*                             $Id: Hypersphere_TopoART.cs 1680 2025-11-15 17:23:15Z marko $                            *
***********************************************************************************************************************/

using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;

namespace LibTopoART
{

//**********************************************************************************************************************

	/// <summary>Class <c>Hypersphere_TopoART</c> provides an implementation of the Hypersphere TopoART neural network
	/// as proposed in "Marko Tscherepanow (2012). Incremental On-line Clustering with a Topology-Learning Hierarchical
	/// ART Neural Network Using Hyperspherical Categories. In Poster and Industry Proceedings of the Industrial
	/// Conference on Data Mining (ICDM), pp. 22–34. Fockendorf, Germany: ibai-publishing."
	/// <para>In contrast to class <c>TopoART</c>, class <c>Hypersphere_TopoART</c> does not require all input to lie in
	/// the interval [0, 1]. The input range is controlled by the radial extend parameter R.</para>
	/// </summary>
	public class Hypersphere_TopoART : TopoART, IHypersphere_TopoART
	{
		private const string _networkName = "Hypersphere TopoART";
		private const NetworkType _networkType = NetworkType.HypersphereTopoART;

//----------------------------------------------------------------------------------------------------------------------

		/// <summary>Property <c>FileFormatVersion</c> returns the version of the file format used by class
		/// <c>Hypersphere_TopoART</c>.</summary>
		public new decimal FileFormatVersion { get => Common.Hypersphere_TopoART_file_format_version; }

		/// <summary>Property <c>HypersphereTopoARTFileFormatVersion</c> returns the version of the file format used by
		/// class <c>Hypersphere_TopoART</c>.</summary>
		public decimal HypersphereTopoARTFileFormatVersion { get => FileFormatVersion; }

//----------------------------------------------------------------------------------------------------------------------

		private TA_F2_node CreateHypersphereTopoARTF2Node(long nodeID,
			long inputLen, decimal[] spatialWeights, long[]? temporalWeights)
		{
			return new HTA_F2_node(nodeID, inputLen, spatialWeights, R);
		}

		private TA_F2_node LoadHypersphereTopoARTF2Node(BinaryReader reader, in (FileFormatVersions fileFormatVersions, bool) fileFormatInfo)
		{
			return new HTA_F2_node(reader, fileFormatInfo.fileFormatVersions, R);
		}

//----------------------------------------------------------------------------------------------------------------------

		/// <value>Property <c>R</c> represents the radial extend parameter R.</value>
		public decimal R { get; private protected set; }

//----------------------------------------------------------------------------------------------------------------------

		// Do not use!
		private protected Hypersphere_TopoART() {}

		/// <summary>This constructor initialises a Hypersphere TopoART network and sets the radial extend parameter to
		/// <c>Math.Sqrt(inputLen)/2</c>.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of Hypersphere TopoART modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first Hypersphere TopoART module (HTA a).</param>
		public Hypersphere_TopoART(long inputLen, long moduleNum, decimal rho_a) :
			this(inputLen, moduleNum, rho_a, (decimal)Math.Sqrt(inputLen) / 2.0m) {}

		/// <summary>This constructor initialises a Hypersphere TopoART network.</summary>
		/// <param name="inputLen">The length of input vectors to be learnt.</param>
		/// <param name="moduleNum">The number of Hypersphere TopoART modules.</param>
		/// <param name="rho_a">The vigilance parameter of the first Hypersphere TopoART module (HTA a).</param>
		/// <param name="R">The radial extend parameter.</param>
		public Hypersphere_TopoART(long inputLen, long moduleNum, decimal rho_a, decimal R)
		{
			SetTopoARTParams(inputLen, moduleNum, rho_a);

			if(R <= 0.0m) {
				this.R = (decimal)Math.Sqrt(inputLen) / 2.0m;
				Common.Warning("Too small value for R, changed to " + this.R);
			} else
				this.R = R;

			Common.Message($"R set to {this.R:0.##########}");

			InitModules(inputLen + 1, CreateTopoARTModule, CreateHypersphereTopoARTF2Node);
		}

		/// <summary>This constructor loads a saved Hypersphere TopoART network.</summary>
		/// <param name="path">The path of a binary Hypersphere TopoART file.</param>
		/// <exception cref="InvalidFileException">Throws when the given file cannot be loaded.</exception>
		public Hypersphere_TopoART(string path)
		{
			using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);

			FileFormatVersions fileFormatVersions = LoadTopoARTParams(file, TopoARTMatchFunction, out var reader);
			InitModules(reader, fileFormatVersions, LoadTopoARTModule, CreateHypersphereTopoARTF2Node, LoadHypersphereTopoARTF2Node);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override decimal[] EncodeCurrentInput()
		{
			decimal[] x_F1;

			Debug.Assert(_x_F0 != null);

			x_F1 = new decimal[_x_F0_len + 1];
			for(long i = 0; i < _x_F0_len; ++i)
				x_F1[i] = _x_F0![i];

			x_F1[_x_F0_len] = 0;

			return x_F1;
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
			R = reader.ReadDecimal();
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void SaveTextHeader(TextWriter writer)
		{
			writer.WriteLine("*********************************");
			writer.WriteLine("*  Hypersphere TopoART network  *");
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
			writer.WriteLine("R: " + R.ToString(CultureInfo.InvariantCulture));
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
			writer.Write(R);
		}

//----------------------------------------------------------------------------------------------------------------------

		private protected override void InitialisationMessage()
		{
			Common.InitialisationMessage(_networkName);
		}
	}

//**********************************************************************************************************************

}