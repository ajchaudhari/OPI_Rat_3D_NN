//Batch Set Slice # and Scale Square Nifty Images to 128x128x[slices]
//By Valerie Porter Chaudhari Lab, UC Davis 5-07-24

//Variables:
//@inputDirectory: This is the folder that contains all MRI T2w Images.
//@outputDirectory: This is the folder where you want to save all of your 
//                  processed images.
//@outputFile: This is the variable that designates the folder location and 
//			   filename for the output file.  It uses the original file name  
//			   w/o ".nii" and adds "_AddSlice64.nii" to the end of it.

run("Clear Results"); // clear the results table of any previous measurements

// The next line prevents ImageJ from showing the processing steps during 
// processing of a large number of images, speeding up the macro
setBatchMode(true); //Enables BatchMode

// Show the user a dialog box to select a directory of images
inputDirectory = getDirectory("Choose Input Directory of Images");
outputDirectory = getDirectory("Choose Output Directory");

// Get the list of files from that directory
// NOTE: if there are non-image files in this directory, it may cause the macro to crash
fileList = getFileList(inputDirectory);

//Run full scaling macro (function is at the bottom)
print("Files whose property values were fixed:"); //This is for the log output file that will be saved.
for (i = 0; i < fileList.length; i++)
{
	if (endsWith(fileList[i],".nii"))
		{
			open(inputDirectory + fileList[i]); //Opens image file
    		//DividebyMax(fileList[i]); //For label images only!!! Converts range to [0,1] (see below)
    		Transformto128(fileList[i]); // Scales image file to 128x128 (see below)
    		SetSlices("1", 64); //Cropping function (see below)
    		outputFile = outputDirectory + substring(fileList[i],0,indexOf(fileList[i], ".nii")) + "_128x128x64.nii"; //New Filename and Folder Location
    		print(outputFile); //Adds name of output file to the log window (for record keeping, optional)
    		selectWindow("1"); //Scaleto128 sets the window name to 1.
    		run("NIfTI-1", "save=[" + outputFile +"]"); //Saves the output file with the new filename
    		close("*"); //closes all windows so that the appropriate window will be selected (see line 28).
		}
}

setBatchMode(false); // Now disable BatchMode since we are finished
selectWindow("Log"); //Cropping function sets the window name to 1.
saveAs("Text", ""+outputDirectory+"PreprocessingLog.txt");
run("Close");

//FOR LABEL IMAGES: Divide by 255 to convert label range to [0,1]
function DividebyMax(LabelFile)
{
	selectWindow(LabelFile);
	depth = nSlices;
	setSlice(depth/2);
	getStatistics(area, mean, min, max, std, histogram);
	//print(max);
	if (max>1)
	{
		run("Divide...", "value="+max+" stack");
	}
	
}

//Setting Number of Slices function: Add slices to MRI along the z-axis to 128x128x[depth], using bicubic averaging
function SetSlices(ImageFile, final_nSlices) //ImageFile: input MRI image; final_nSlice: total number of desired slices
{
	selectWindow(ImageFile); //Cropping function sets the window name to 1.
	original_nSlices = nSlices;
	
	//if images has too few slices, addes slices at the back of the image volume
	if (original_nSlices < final_nSlices)
	{
		for(i=0; i < final_nSlices-original_nSlices; i++) //adds (final_nSlices-original_nSlices) slices to image volume 
		{	
			depth = nSlices; //detects the number of slices
			setSlice(depth); //sets the image viewer to the "depth" slice
			run("Add Slice"); //adds blank slice behind selected slice
		}
	}
	
	//if image has too many slices, removes slices from the back to front
	if (original_nSlices > final_nSlices) 
	{
		for(i=0; i < original_nSlices-final_nSlices; i++) //removes (final_nSlices-original_nSlices) slices from image volume 
		{
			depth = nSlices; //detects the number of slices; set this 1 if you want the first slice or start from the front of the image!
			setSlice(depth); //sets the image viewer to the "depth" slice
			run("Delete Slice"); //adds blank slice behind selected slice
		}
	}
	else 
	{
	
	}
}

//Cropping function: Down scales square image to 128x128
function Transformto128(ImageFile)
{
	//Making a square image:
	//Comment out with // at the beginning of the line to remove from code.
	selectWindow(ImageFile); //Image must be square!!! [m x m]
	makeRectangle(40, 0, 200, 200); //Makes a 200x200x(number of slices) voxels ROI that is centered in the x-direction
	roiManager("Add"); //Adds ROI to roi manager
	run("Crop"); //Crops 280x200 image to a 200x200 image
	
	//Downscaling the square image to 128x128 pixels:
	//**Be sure to add // to line 113 if you use line 109 below!
	run("Scale...", "x=- y=- z=1.0 width=128 height=128 interpolation=Bicubic average process create title=1"); //For images...

	//FOR LABEL IMAGES (do not use interpolations): Uncomment by deleting //
	//**Be sure to add // to line 109 if you use line 113 below!
	//run("Scale...", "x=- y=- z=1.0 width=128 height=128 depth="+nSlices+" interpolation=None process create title=1"); //For labels...

	//Extra commands for image pre-processing, if needed.
	//To Flip the Z-axis (back-to-front):
	//run("Flip Z"); //Can comment out if not needed

	//To rotate image (specify angle in degrees):
	//run("Rotate... ", "angle=180 grid=1 interpolation=None stack");
}
