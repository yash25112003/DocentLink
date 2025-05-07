const validateProfile = (req, res, next) => {
  const { university, userDetails, professors } = req.body;

  // Validate university
  if (!university || !university.name) {
    return res.status(400).json({ error: 'University name is required' });
  }

  const validUniversities = [
    "Arizona State University",
    "Stanford University",
    "MIT",
    "Harvard University",
    "UC Berkeley"
  ];

  if (!validUniversities.includes(university.name)) {
    return res.status(400).json({ error: 'Invalid university name' });
  }

  // Validate user details
  if (!userDetails || !userDetails.name || !userDetails.email) {
    return res.status(400).json({ error: 'User name and email are required' });
  }

  if (userDetails.name.length < 2 || userDetails.name.length > 100) {
    return res.status(400).json({ error: 'Name must be between 2 and 100 characters' });
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(userDetails.email)) {
    return res.status(400).json({ error: 'Invalid email format' });
  }

  // Validate professors
  if (!Array.isArray(professors) || professors.length === 0 || professors.length > 10) {
    return res.status(400).json({ error: 'Must select between 1 and 10 professors' });
  }

  for (const prof of professors) {
    if (!prof.name || !prof.sourceUniversity) {
      return res.status(400).json({ error: 'Each professor must have a name and source university' });
    }
  }

  next();
};

module.exports = {
  validateProfile
}; 