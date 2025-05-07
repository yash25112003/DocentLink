const express = require('express');
const router = express.Router();
const UserProfile = require('../models/UserProfile');
const { validateProfile } = require('../middlewares/validation');

// Create a new profile
router.post('/', validateProfile, async (req, res, next) => {
  try {
    const profile = new UserProfile({
      university: req.body.university,
      userDetails: req.body.userDetails,
      selectedProfessors: req.body.professors.map(prof => ({
        name: prof.name,
        sourceUniversity: prof.sourceUniversity
      })),
      metadata: {
        ipAddress: req.ip,
        userAgent: req.headers['user-agent']
      }
    });

    await profile.save();
    res.status(201).json(profile);
  } catch (err) {
    next(err);
  }
});

// Get profile by session ID
router.get('/:sessionId', async (req, res, next) => {
  try {
    const profile = await UserProfile.findOne({ sessionId: req.params.sessionId });
    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' });
    }
    res.json(profile);
  } catch (err) {
    next(err);
  }
});

// Get profiles by university
router.get('/university/:universityName', async (req, res, next) => {
  try {
    const profiles = await UserProfile.findByUniversity(req.params.universityName);
    res.json(profiles);
  } catch (err) {
    next(err);
  }
});

module.exports = router; 