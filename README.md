# SWSharp

(Vibe coded with Pi: https://github.com/badlogic/pi-mono)

Swift/Metal port of SHARP ([Paper](https://arxiv.org/abs/2512.10685)/[GitHub](https://github.com/apple/ml-sharp)) - Monocular 3D Gaussian Prediction.

Takes a single image, outputs 3D Gaussian splats (.ply) in ~3 seconds.

## Model Setup

The model must be converted from PyTorch to CoreML:

```bash
# Download and convert (requires ml-sharp repo: https://github.com/apple/ml-sharp.git)
just setup-model /path/to/ml-sharp

# Or step by step:
just download-model                    # Download .pt from Apple CDN
just convert-model /path/to/ml-sharp   # Convert to .mlpackage
just compile-model                     # Compile to .mlmodelc (faster)
```

Get ml-sharp:
```bash
git clone https://github.com/apple/ml-sharp.git
```

## Usage

```bash
# Single image
sharp-cli -i image.jpg -m Models/SharpPredictor.mlmodelc

# Directory of images
sharp-cli -i photos/ -m Models/SharpPredictor.mlmodelc

# Custom output directory
sharp-cli -i image.jpg -o output/ -m Models/SharpPredictor.mlmodelc

# Verbose
sharp-cli -i image.jpg -m Models/SharpPredictor.mlmodelc -v
```

## License

### Sharp (This Project)

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Apple ml-sharp (Original Implementation)

This project is a port of Apple's [ml-sharp](https://github.com/apple/ml-sharp). The original code is licensed under the following terms:

<details>
<summary>Apple Software License</summary>

```
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple
Inc. ("Apple") in consideration of your agreement to the following
terms, and your use, installation, modification or redistribution of
this Apple software constitutes acceptance of these terms.  If you do
not agree with these terms, please do not use, install, modify or
redistribute this Apple software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, Apple grants you a personal, non-exclusive
license, under Apple's copyrights in this original Apple software (the
"Apple Software"), to use, reproduce, modify and redistribute the Apple
Software, with or without modifications, in source and/or binary forms;
provided that if you redistribute the Apple Software in its entirety and
without modifications, you must retain this notice and the following
text and disclaimers in all such redistributions of the Apple Software.
Neither the name, trademarks, service marks or logos of Apple Inc. may
be used to endorse or promote products derived from the Apple Software
without specific prior written permission from Apple.  Except as
expressly stated in this notice, no other rights or licenses, express or
implied, are granted by Apple herein, including but not limited to any
patent rights that may be infringed by your derivative works or by other
works in which the Apple Software may be incorporated.

The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.

IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

</details>

### Apple SHARP Model

The SHARP model weights are licensed separately under Apple's Machine Learning Research Model License:

<details>
<summary>Apple Machine Learning Research Model License</summary>

```
Disclaimer: IMPORTANT: This Apple Machine Learning Research Model is
specifically developed and released by Apple Inc. ("Apple") for the sole purpose
of scientific research of artificial intelligence and machine-learning
technology. "Apple Machine Learning Research Model" means the model, including
but not limited to algorithms, formulas, trained model weights, parameters,
configurations, checkpoints, and any related materials (including
documentation).

This Apple Machine Learning Research Model is provided to You by
Apple in consideration of your agreement to the following terms, and your use,
modification, creation of Model Derivatives, and or redistribution of the Apple
Machine Learning Research Model constitutes acceptance of this Agreement. If You
do not agree with these terms, please do not use, modify, create Model
Derivatives of, or distribute this Apple Machine Learning Research Model or
Model Derivatives.

* License Scope: In consideration of your agreement to abide by the following
  terms, and subject to these terms, Apple hereby grants you a personal,
  non-exclusive, worldwide, non-transferable, royalty-free, revocable, and
  limited license, to use, copy, modify, distribute, and create Model
  Derivatives (defined below) of the Apple Machine Learning Research Model
  exclusively for Research Purposes. You agree that any Model Derivatives You
  may create or that may be created for You will be limited to Research Purposes
  as well. "Research Purposes" means non-commercial scientific research and
  academic development activities, such as experimentation, analysis, testing
  conducted by You with the sole intent to advance scientific knowledge and
  research. "Research Purposes" does not include any commercial exploitation,
  product development or use in any commercial product or service.

* Distribution of Apple Machine Learning Research Model and Model Derivatives:
  If you choose to redistribute Apple Machine Learning Research Model or its
  Model Derivatives, you must provide a copy of this Agreement to such third
  party, and ensure that the following attribution notice be provided: "Apple
  Machine Learning Research Model is licensed under the Apple Machine Learning
  Research Model License Agreement." Additionally, all Model Derivatives must
  clearly be identified as such, including disclosure of modifications and
  changes made to the Apple Machine Learning Research Model. The name,
  trademarks, service marks or logos of Apple may not be used to endorse or
  promote Model Derivatives or the relationship between You and Apple. "Model
  Derivatives" means any models or any other artifacts created by modifications,
  improvements, adaptations, alterations to the architecture, algorithm or
  training processes of the Apple Machine Learning Research Model, or by any
  retraining, fine-tuning of the Apple Machine Learning Research Model.

* No Other License: Except as expressly stated in this notice, no other rights
  or licenses, express or implied, are granted by Apple herein, including but
  not limited to any patent, trademark, and similar intellectual property rights
  worldwide that may be infringed by the Apple Machine Learning Research Model,
  the Model Derivatives or by other works in which the Apple Machine Learning
  Research Model may be incorporated.

* Compliance with Laws: Your use of Apple Machine Learning Research Model must
  be in compliance with all applicable laws and regulations.

* Term and Termination: The term of this Agreement will begin upon your
  acceptance of this Agreement or use of the Apple Machine Learning Research
  Model and will continue until terminated in accordance with the following
  terms. Apple may terminate this Agreement at any time if You are in breach of
  any term or condition of this Agreement. Upon termination of this Agreement,
  You must cease to use all Apple Machine Learning Research Models and Model
  Derivatives and permanently delete any copy thereof. Sections 3, 6 and 7 will
  survive termination.

* Disclaimer and Limitation of Liability: This Apple Machine Learning Research
  Model and any outputs generated by the Apple Machine Learning Research Model
  are provided on an "AS IS" basis. APPLE MAKES NO WARRANTIES, EXPRESS OR
  IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE,
  REGARDING THE APPLE MACHINE LEARNING RESEARCH MODEL OR OUTPUTS GENERATED BY
  THE APPLE MACHINE LEARNING RESEARCH MODEL. You are solely responsible for
  determining the appropriateness of using or redistributing the Apple Machine
  Learning Research Model and any outputs of the Apple Machine Learning Research
  Model and assume any risks associated with Your use of the Apple Machine
  Learning Research Model and any output and results. IN NO EVENT SHALL APPLE BE
  LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING
  IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION OF
  THE APPLE MACHINE LEARNING RESEARCH MODEL AND ANY OUTPUTS OF THE APPLE MACHINE
  LEARNING RESEARCH MODEL, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT,
  TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS
  BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

* Governing Law: This Agreement will be governed by and construed under the laws
  of the State of California without regard to its choice of law principles. The
  Convention on Contracts for the International Sale of Goods shall not apply to
  the Agreement except that the arbitration clause and any arbitration hereunder
  shall be governed by the Federal Arbitration Act, Chapters 1 and 2.

Copyright (C) 2025 Apple Inc. All Rights Reserved.
```

</details>

