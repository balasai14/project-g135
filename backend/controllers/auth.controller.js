import bcryptjs from "bcryptjs";
import { User } from "../models/user.model.js";
import { generateTokenAndSetCookie } from "../utils/generateTokenAndSetCookie.js";


export const signup = async (req, res) => {
	const { email, password, name, image } = req.body;
  
	try {
	  if (!email || !password || !name || !image) {
		throw new Error("All fields, including an image, are required");
	  }
  
	  // Check if the user already exists
	  const userAlreadyExists = await User.findOne({ email });
	  console.log("userAlreadyExists", userAlreadyExists);
  
	  if (userAlreadyExists) {
		return res.status(400).json({
		  success: false,
		  message: "User already exists",
		});
	  }
  
	  // Hash the password
	  const hashedPassword = await bcryptjs.hash(password, 10);
	  const verificationToken = Math.floor(100000 + Math.random() * 900000).toString();
  
	  // Convert the base64 image to a Buffer
	  const imageBuffer = Buffer.from(image.split(",")[1], "base64");
  
	  // Create a new user
	  const user = new User({
		email,
		password: hashedPassword,
		name,
		isVerified: true, // Automatically verified for simplicity
		image: imageBuffer, // Store the user's image
	  });
  
	  await user.save();
  
	  // Generate JWT and set cookie
	  generateTokenAndSetCookie(res, user._id);
  
	  res.status(201).json({
		success: true,
		message: "User created successfully",
		user: {
		  ...user._doc,
		  password: undefined, // Exclude password from the response
		},
	  });
	} catch (error) {
	  res.status(400).json({
		success: false,
		message: error.message,
	  });
	}
};
export const login = async (req, res) => {
	const { email, password } = req.body;
	try {
		const user = await User.findOne({ email });
		if (!user) {
			return res.status(400).json({ success: false, message: "Invalid credentials" });
		}
		const isPasswordValid = await bcryptjs.compare(password, user.password);
		if (!isPasswordValid) {
			return res.status(400).json({ success: false, message: "Invalid credentials" });
		}

		generateTokenAndSetCookie(res, user._id);

		user.lastLogin = new Date();
		await user.save();

		res.status(200).json({
			success: true,
			message: "Logged in successfully",
			user: {
				...user._doc,
				password: undefined,
			},
		});
	} catch (error) {
		console.log("Error in login ", error);
		res.status(400).json({ success: false, message: error.message });
	}
};
export const logout = async (req, res) => {
	res.clearCookie("token");
	res.status(200).json({ success: true, message: "Logged out successfully" });
};
export const checkAuth = async (req, res) => {
    try {
        const user = await User.findById(req.userId).select("-password");

        if (!user) {
            return res.status(400).json({ success: false, message: "User not found" });
        }

        
        res.status(200).json({ success: true, user });
    } catch (error) {
        console.log("Error in checkAuth ", error);
        res.status(400).json({ success: false, message: error.message });
    }
};


